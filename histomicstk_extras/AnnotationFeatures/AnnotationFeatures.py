import json
import os
import pprint
import random
import time
from pathlib import Path

import girder_client
import histomicstk.features
import large_image
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
import rasterio.features
import shapely
import sklearn.cluster
import sklearn.manifold
import sklearn.metrics
import umap
from histomicstk.cli import utils as cli_utils
from histomicstk.cli.utils import CLIArgumentParser


def annotation_to_shapely(annot, offset=(0, 0)):
    return shapely.polygons([
        shapely.linearrings([[p[0] - offset[0], p[1] - offset[1]]
                            for p in element['points']])
        for element in annot['annotation']['elements']
        if element['type'] == 'polyline' and element['closed']
    ])


def compute_tile(ts, tile, polys, frame, bidx, cyto_width):
    if not len(polys):
        return None
    if tile['tile'].shape[-1] in {2, 4} and bidx + 1 == tile['tile'].shape[-1]:
        return None
    polys = [(shapely.affinity.translate(pp, -tile['x'], -tile['y']), idx)
             for pp, idx in polys]
    labels = rasterio.features.rasterize(polys, out_shape=(tile['height'], tile['width']))
    print('Getting features for tile '
          f'{tile["tile_position"]["position"]}, frame {frame}, band {bidx}')
    try:
        df = histomicstk.features.compute_nuclei_features(
            labels,
            tile['tile'][:, :, bidx],
            tile['tile'][:, :, bidx] if cyto_width else None,
            morphometry_features_flag=not bidx and not frame,
            cyto_width=cyto_width)
    except Exception as exc:
        print(f'Failed {exc}')
        import traceback
        print(traceback.format_exc())
        return None
    df = df.replace('NaN', 0)
    df.fillna(0, inplace=True)
    df = df.drop(df[
        df['Identifier.Xmin'] > tile['width'] - tile['tile_overlap']['right']].index)
    df = df.drop(df[
        df['Identifier.Ymin'] > tile['height'] - tile['tile_overlap']['bottom']].index)
    df = df.drop(df[
        df['Identifier.Xmax'] < tile['tile_overlap']['left']].index)
    df = df.drop(df[
        df['Identifier.Ymax'] < tile['tile_overlap']['right']].index)
    # Adjust these for tile position
    for key in {'Identifier.Xmin', 'Identifier.Xmax',
                'Identifier.CentroidX', 'Identifier.WeightedCentroidX'}:
        df[key] += tile['x']
    for key in {'Identifier.Ymin', 'Identifier.Ymax',
                'Identifier.CentroidY', 'Identifier.WeightedCentroidY'}:
        df[key] += tile['y']
    df = df.rename(columns={
        key: (f'{key}'
              f'{(".band" + str(bidx)) if tile["tile"].shape[-1] > 1 else ""}'
              f'{(".frame" + str(frame)) if ts.frames > 1 else ""}')
        for key in df.columns.tolist()
        if key.startswith(('Nucl', 'Cyto'))})
    return df


def process(ts, annot, args):  # noqa
    import dask

    indices = []
    tasks = []
    # iterate over frames
    basepolys = annotation_to_shapely(annot, (0, 0))
    bands = None
    for frame in range(ts.frames):
        for tile in ts.tileIterator(
                tile_size=dict(width=4096, height=4096),
                tile_overlap=dict(x=512, y=512),
                frame=frame):
            print(tile)
            tilebox = shapely.geometry.box(
                tile['x'], tile['y'], tile['x'] + tile['width'], tile['y'] + tile['height'])
            polys = [(pp, idx + 1) for idx, pp in enumerate(basepolys) if pp.intersects(tilebox)]
            if bands is None:
                bands = tile['tile'].shape[-1]
                if bands in {2, 4}:
                    bands -= 1
                tile.release()
            # for each band
            for bidx in range(bands):
                indices.append((frame, tile['tile_position']['position'], bidx))
                tasks.append(dask.delayed(compute_tile)(
                    ts, tile, polys, frame, bidx, args.cyto_width))
    tasks = dask.delayed(tasks).compute()
    tiledfs = {}
    for (frame, tileidx, _bidx), df in zip(indices, tasks):
        if df is None:
            continue
        key = (frame, tileidx)
        if key not in tiledfs:
            tiledfs[key] = df
        else:
            df = df.drop(columns={
                key for key in df.columns
                if not key == 'Label' and
                not key.startswith('Nucl') and
                not key.startswith('Cyto')})
            tiledfs[key] = tiledfs[key].merge(
                df, how='outer', on=list(set(tiledfs[key].columns) & set(df.columns)))
    framedfs = {}
    for (frame, _tileidx), tiledf in tiledfs.items():
        if tiledf is None:
            continue
        if frame not in framedfs:
            framedfs[frame] = tiledf
        else:
            df = tiledf.drop(tiledf[tiledf['Label'].isin(framedfs[frame]['Label'])].index)
            framedfs[frame] = pd.concat((framedfs[frame], df))
    totaldf = None
    for _frame, framedf in framedfs.items():
        if framedf is None:
            continue
        if totaldf is None:
            totaldf = framedf
        else:
            framedf = framedf.drop(columns={
                key for key in framedf.columns
                if not key == 'Label' and
                not key.startswith('Nucl') and
                not key.startswith('Cyto')})
            totaldf = totaldf.merge(
                framedf, how='outer', on=list(set(totaldf.columns) & set(framedf.columns)))
        if framedf is not None and totaldf is not None:
            print(f'Frame data {framedf.shape}, collected {totaldf.shape}')
    print(f'{len(list(totaldf["Label"]))} elements')
    totaldf[['Label']] = totaldf[['Label']].astype(int) - 1
    totaldf.reset_index()
    print(totaldf)
    print(totaldf.columns.tolist())
    return totaldf


def df_process(totaldf, annot, args):
    start_time = time.time()
    totaldf.to_csv(args.featureFile, index=False)
    print(f'Writing csv time {cli_utils.disp_time_hms(time.time() - start_time)}')
    start_time = time.time()
    df = totaldf.drop(columns={
        key for key in totaldf.columns if key.startswith(('Label', 'Identifier'))})
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(df)
    scaleddf = scaler.transform(df)
    print(f'Transforming data for fit {cli_utils.disp_time_hms(time.time() - start_time)}')
    start_time = time.time()
    # Calculate kmeans; if clusters is 0, guess
    if args.clusters < 2:
        lastinertia = None
        firstInertiaSlope = None
        bestn = 2
        for n_clusters in range(2, 21):
            clusterer = sklearn.cluster.KMeans(
                n_clusters=n_clusters, init='k-means++', max_iter=100)
            kmeans_labels = clusterer.fit_predict(scaleddf)
            inertia = clusterer.inertia_
            if lastinertia is not None:
                inertiaSlope = inertia - lastinertia
                print('inertiaSlope', inertiaSlope)
                if firstInertiaSlope is None:
                    firstInertiaSlope = inertiaSlope
                elif abs(inertiaSlope) < abs(firstInertiaSlope / 4):
                    break
            bestn = n_clusters
            lastinertia = inertia
            print('KMeans clustering total time '
                  f'{cli_utils.disp_time_hms(time.time() - start_time)}')
        print(f'KMeans clustering total time with {bestn} clusters '
              f'{cli_utils.disp_time_hms(time.time() - start_time)}')
    else:
        print(f'Doing k-means with {args.clusters} clusters.')
        kmeans_labels = sklearn.cluster.KMeans(
            n_clusters=args.clusters, init='k-means++', max_iter=100).fit_predict(scaleddf)
        print(f'KMeans clustering total time {cli_utils.disp_time_hms(time.time() - start_time)}')
    print(kmeans_labels)
    start_time = time.time()
    # Add the cluster labels to the data we use with umap/tsne
    df['Cluster'] = kmeans_labels
    fit = umap.UMAP(
        min_dist=0.01,
        n_neighbors=5,
        # metric='wminkowski',
    )
    umapVal = fit.fit_transform(df)
    print(f'UMAP time {cli_utils.disp_time_hms(time.time() - start_time)}')
    print(umapVal)
    start_time = time.time()
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
    tsneVal = tsne.fit_transform(df)
    print(f'TSNE time {cli_utils.disp_time_hms(time.time() - start_time)}')
    print(tsneVal)
    start_time = time.time()
    # We want to add umap and cluster labels to dataframe and resave
    totaldf['Cluster'] = kmeans_labels
    totaldf['umap.x'] = umapVal[:, 0]
    totaldf['umap.y'] = umapVal[:, 1]
    totaldf['tsne.x'] = tsneVal[:, 0]
    totaldf['tsne.y'] = tsneVal[:, 1]

    # outputNucleiAnnotationFile
    viridis = [
        '#440154', '#482172', '#423d84', '#38578c', '#2d6f8e', '#24858d',
        '#1e9a89', '#2ab07e', '#51c468', '#86d449', '#c2df22', '#fde724']
    colorBrewerPaired12 = [
        '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
        '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
    colors = colorBrewerPaired12 + viridis + [
        matplotlib.colors.rgb2hex(c) for c in plt.get_cmap('tab20c').colors]
    colorsa = [f'rgba({int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:6], 16)}, 0.5)'
               for c in colors]
    elements = []
    mlist = []
    meta = {'features': mlist}
    for _, row in totaldf.iterrows():
        element = annot['annotation']['elements'][int(row['Label'])]
        element.pop('id', None)
        element.pop('_id', None)
        element['id'] = '%024x' % random.randrange(16**24)
        # add cluster, umap, tsne, change color based on cluster
        element['user'] = {
            key: row[key] for key in {
                'Cluster', 'umap.x', 'umap.y', 'tsne.x', 'tsne.y',
            }}
        element['lineColor'] = colors[int(row['Cluster']) % len(colors)]
        element['fillColor'] = colorsa[int(row['Cluster']) % len(colors)]
        elements.append(element)
        mentry = {'id': element['id']}
        mentry.update(element['user'])
        mlist.append(mentry)
    annot_fname = os.path.splitext(
        os.path.basename(args.outputAnnotationFile))[0]
    annotation = {
        'name': annot_fname + '-features',
        'elements': elements,
        'attributes': {
            'params': vars(args),
            'cli': Path(__file__).stem,
        },
    }
    print(f'Prepare for save time {cli_utils.disp_time_hms(time.time() - start_time)}')
    start_time = time.time()
    with open(args.outputAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)
    print(f'Save annotation time {cli_utils.disp_time_hms(time.time() - start_time)}')
    start_time = time.time()
    with open(args.outputItemMetadata, 'w') as metadata_file:
        json.dump(meta, metadata_file, separators=(',', ':'), sort_keys=False)
    print(f'Save metadata time {cli_utils.disp_time_hms(time.time() - start_time)}')


def main(args):
    print('\n>> CLI Parameters ...\n')
    pprint.pprint(vars(args))
    if not args.style or args.style.startswith('{#control'):
        args.style = None
    ts = large_image.open(args.image, style=args.style)
    pprint.pprint(ts.metadata)
    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.token = args.girderToken
    annot = gc.get(f'annotation/{args.annotationID.strip()}')
    print(f'Annotation has {len(annot["annotation"]["elements"])} elements')
    start_time = time.time()
    c = cli_utils.create_dask_client(args)
    print(c)
    dask_setup_time = time.time() - start_time
    print(f'Dask setup time = {cli_utils.disp_time_hms(dask_setup_time)}')
    totaldf = process(ts, annot, args)
    c.shutdown()
    c.close()
    print(f'Base processing time {cli_utils.disp_time_hms(time.time() - start_time)}')
    df_process(totaldf, annot, args)
    print(f'Total time {cli_utils.disp_time_hms(time.time() - start_time)}')


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
