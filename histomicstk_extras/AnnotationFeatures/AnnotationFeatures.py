import json
import os
import pprint
import random
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
from histomicstk.cli.utils import CLIArgumentParser


def annotation_to_shapely(annot, offset=(0, 0)):
    return shapely.polygons([
        shapely.linearrings([[p[0] - offset[0], p[1] - offset[1]]
                            for p in element['points']])
        for element in annot['annotation']['elements']
        if element['type'] == 'polyline' and element['closed']
    ])


def main(args):  # noqa
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
    totaldf = None
    # iterate over frames
    for frame in range(ts.frames):
        framedf = None
        for tile in ts.tileIterator(
                tile_size=dict(width=4096, height=4096),
                tile_overlap=dict(x=512, y=512),
                frame=frame):
            print(tile)
            labels = rasterio.features.rasterize(
                [(pp, idx + 1) for idx, pp in enumerate(
                    annotation_to_shapely(annot, (tile['x'], tile['y'])))],
                out_shape=(tile['height'], tile['width']))
            tiledf = None
            # for each band
            for bidx in range(tile['tile'].shape[-1]):
                if tile['tile'].shape[-1] in {2, 4} and bidx + 1 == tile['tile'].shape[-1]:
                    continue
                print('Getting features for tile '
                      f'{tile["tile_position"]["position"]}, frame {frame}, band {bidx}')
                try:
                    df = histomicstk.features.compute_nuclei_features(
                        labels,
                        tile['tile'][:, :, bidx],
                        tile['tile'][:, :, bidx] if args.cyto_width else None,
                        morphometry_features_flag=not bidx and not frame,
                        cyto_width=args.cyto_width)
                except Exception as exc:
                    print(f'Failed {exc}')
                    import traceback
                    print(traceback.format_exc())
                    continue
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
                if tiledf is None:
                    tiledf = df
                elif df is not None:
                    df = df.drop(columns={
                        key for key in df.columns
                        if not key == 'Label' and
                        not key.startswith('Nucl') and
                        not key.startswith('Cyto')})
                    tiledf = tiledf.merge(
                        df, how='outer', on=list(set(tiledf.columns) & set(df.columns)))
                if df is not None and tiledf is not None:
                    print(f'Band data {df.shape}, collected {tiledf.shape}')
            print(tiledf)
            if tiledf is not None:
                print(tiledf.columns.tolist())
            if framedf is None:
                framedf = tiledf
            elif tiledf is not None:
                df = tiledf.drop(tiledf[tiledf['Label'].isin(framedf['Label'])].index)
                framedf = pd.concat((framedf, df))
            if tiledf is not None and framedf is not None:
                print(f'Tile data {tiledf.shape}, collected {framedf.shape}')
        print(framedf)
        if framedf is not None:
            print(framedf.columns.tolist())
        if totaldf is None:
            totaldf = framedf
        elif framedf is not None:
            framedf = framedf.drop(columns={
                key for key in framedf.columns
                if not key == 'Label' and
                not key.startswith('Nucl') and
                not key.startswith('Cyto')})
            totaldf = totaldf.merge(
                framedf, how='outer', on=list(set(totaldf.columns) & set(framedf.columns)))
        if framedf is not None and totaldf is not None:
            print(f'Frame data {framedf.shape}, collected {totaldf.shape}')
    print(list(totaldf['Label']))
    totaldf[['Label']] = totaldf[['Label']].astype(int) - 1
    totaldf.reset_index()
    print(totaldf)
    print(totaldf.columns.tolist())
    totaldf.to_csv(args.featureFile, index=False)
    # Calculate kmeans; if clusters is 0, guess
    if args.clusters < 2:
        lastsil = 10
        bestn = 2
        for n_clusters in range(2, 21):
            model = sklearn.cluster.KMeans(
                n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)
            labels = model.fit_predict(totaldf)
            sil_score = sklearn.metrics.silhouette_score(totaldf, labels)
            print('The average silhouette score for %i clusters is %0.4f' % (n_clusters, sil_score))
            if sil_score > lastsil:
                bestn = n_clusters
                break
            lastsil = sil_score
    else:
        bestn = args.clusters
    print(f'Doing k-means with {bestn} clusters.')
    kmeans_labels = sklearn.cluster.KMeans(n_clusters=bestn).fit_predict(totaldf)
    print(kmeans_labels)
    fit = umap.UMAP()
    umapVal = fit.fit_transform(totaldf)
    print(umapVal)
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
    tsneVal = tsne.fit_transform(totaldf)
    print(tsneVal)
    # We want to add umap and cluster labels to dataframe and resave
    totaldf['Cluster'] = kmeans_labels
    totaldf['umap.x'] = umapVal[:, 0]
    totaldf['umap.y'] = umapVal[:, 1]
    totaldf['tsne.x'] = tsneVal[:, 0]
    totaldf['tsne.y'] = tsneVal[:, 1]

    # outputNucleiAnnotationFile
    colors = [matplotlib.colors.rgb2hex(c) for c in plt.get_cmap('tab20c').colors]
    colorsa = [f'rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, 0.25)'
               for c in plt.get_cmap('tab20c').colors]
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
    with open(args.outputAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)
    with open(args.outputItemMetadata, 'w') as metadata_file:
        json.dump(meta, metadata_file, separators=(',', ':'), sort_keys=False)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
