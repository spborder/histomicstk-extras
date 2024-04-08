import copy
import logging
import math
import os
import pprint
import sys
import tempfile

import girder_client
import histomicstk
import large_image
import large_image_converter
import numpy as np
import pystackreg
import rasterio.features
import shapely
import skimage.filters
import skimage.morphology
import skimage.transform
import yaml
from histomicstk.cli.utils import CLIArgumentParser
from histomicstk.preprocessing.color_deconvolution.stain_color_map import \
    stain_color_map

from progress_helper import ProgressHelper


def annotation_to_shapely(annot, reduce=1):
    return shapely.polygons([
        shapely.linearrings([[p[0] / reduce, p[1] / reduce]
                            for p in element['points']])
        for element in annot['annotation']['elements']
        if element['type'] == 'polyline' and element['closed']
    ])


def get_image(ts, sizeX, sizeY, frame, annotID, args, reduce):
    regionparams = {'format': large_image.constants.TILE_FORMAT_NUMPY}
    try:
        regionparams['frame'] = int(frame)
    except Exception:
        try:
            if 'channelmap' in ts.metadata and 'DAPI' in ts.metadata['channelmap']:
                regionparams['frame'] = (
                    ts.metadata['channelmap']['DAPI'] * ts.metadata['IndexStride']['IndexC'])
        except Exception:
            pass
    if annotID:
        gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
        gc.token = args.girderToken
        annot = gc.get(f'annotation/{annotID.strip()}')
        if annot['annotation']['elements'][0]['type'] == 'point':
            return annot['annotation']['elements']
        img = (rasterio.features.rasterize(
            annotation_to_shapely(annot), out_shape=(sizeY, sizeX)) > 0).astype('bool')
        img = 255.0 * img
    else:
        regionparams['output'] = dict(maxWidth=ts.sizeX // reduce, maxHeight=ts.sizeY // reduce)
        img = ts.getRegion(**regionparams)[0]
        print(f'Region shape {img.shape}')
        if len(img.shape) == 3 and img.shape[-1] >= 3:
            img = img[:, :, :3]
            # possibly get from args
            stains = ['hematoxylin', 'eosin', 'null']
            print('Deconvolving')
            stain_matrix = np.array([stain_color_map[stain] for stain in stains]).T
            img = histomicstk.preprocessing.color_deconvolution.color_deconvolution(
                img, stain_matrix).Stains[:, :, 0]
            img = 255 - img
        elif len(img.shape) == 3:
            print('Using directly')
            img = img[:, :, 0]
        img = np.pad(img, ((0, sizeY - img.shape[0]), (0, sizeX - img.shape[1])), mode='constant')

        if args.threshold:
            print('Thresholding')
            img = (img > skimage.filters.threshold_otsu(img))
            if args.smallObject:
                img = skimage.morphology.remove_small_objects(img, args.smallObject)
            if args.disk:
                img = skimage.morphology.binary_opening(img, skimage.morphology.disk(args.disk))
            if args.smallObject:
                img = skimage.morphology.remove_small_objects(img, args.smallObject)
            img = 255.0 * img
    return img


def transform_images(ts1, ts2, matrix, out2path=None, outmergepath=None):
    if hasattr(matrix, 'tolist'):
        matrix = matrix.tolist()
    trans2 = {
        'name': f'Transform of {os.path.basename(ts2.largeImagePath)}',
        'width': max(ts1.sizeX, ts2.sizeX),
        'height': max(ts1.sizeY, ts2.sizeY),
        'backgroundColor': [0, 0, 0],
        'scale': {},
        'sources': [{
            'path': ts2.largeImagePath,
            'position': {
                's11': matrix[0][0],
                's12': matrix[0][1],
                'x': matrix[0][2],
                's21': matrix[1][0],
                's22': matrix[1][1],
                'y': matrix[1][2],
            },
        }],
    }
    if ts2.frames > 1:
        trans2['singleBand'] = True
    for k in {'mm_x', 'mm_y', 'magnfication'}:
        val = ts2.metadata.get(k) or ts1.metadata.get(k) or 0
        if val:
            trans2['scale'][k] = val
    print('---')
    print(yaml.dump(trans2, sort_keys=False))

    combo = copy.deepcopy(trans2)
    combo['name'] = (
        f'Transform of {os.path.basename(ts2.largeImagePath)} with '
        f'{os.path.basename(ts1.largeImagePath)}')
    if ts1.frames == 1 and ts2.frames == 1:
        combo['sources'].append({
            'path': ts1.largeImagePath,
            'z': 1,
        })
    elif ts1.frames == 1 and ts2.frames > 1:
        for band in range(1 if ts1.bandCount < 3 else 3):
            combo['sources'].append({
                'path': ts1.largeImagePath,
                'channel': ['red', 'green', 'blue'][band] if ts1.bandCount >= 3 else 'gray',
                'z': 0,
                'c': band + ts2.frames,
                'style': {'dtype': 'uint8', 'bands': [{'band': band + 1, 'palette': 'white'}]},
            })
    else:
        combo['singleBand'] = True
        if ts2.frames == 1:
            src = combo['sources'].pop()
            for band in range(1, 1 if ts1.bandCount < 3 else 3):
                src.update({
                    'channel': ['red', 'green', 'blue'][band] if ts2.bandCount >= 3 else 'gray',
                    'z': 0,
                    'c': band,
                    'style': {'dtype': 'uint8', 'bands': [{'band': band + 1, 'palette': 'white'}]},
                })
                combo['sources'].append(src)
        else:
            combo['sources'].append({
                'path': ts1.largeImagePath,
                'z': 0,
                'c': len(combo['sources']),
            })
    print('---')
    print(yaml.dump(combo, sort_keys=False))
    print('\n---')
    with tempfile.TemporaryDirectory() as tmpdir:
        sys.stdout.flush()
        logger = logging.getLogger('large_image')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger = logging.getLogger('large-image-converter')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        if out2path:
            trans2path = os.path.join(tmpdir, 'out2transform.yaml')
            open(trans2path, 'w').write(yaml.dump(trans2))
            large_image_converter.convert(trans2path, out2path)
        if outmergepath:
            combopath = os.path.join(tmpdir, 'outmergetransform.yaml')
            open(combopath, 'w').write(yaml.dump(combo))
            large_image_converter.convert(combopath, outmergepath)


def register_points(args, points1, points2):
    if not isinstance(points1, list) or not isinstance(points2, list):
        msg = 'If one annotation is points, both images must specify a point annotation'
        raise Exception(msg)
    labels = {}
    allpts = []
    for key, points in [('1', points1), ('2', points2)]:
        for idx, pt in enumerate(points):
            if len(allpts) < idx + 1:
                allpts.append({})
            allpts[idx][key] = [pt['center'][0], pt['center'][1]]
            label = pt.get('label', {}).get('value')
            if label:
                labels.setdefault(label, {})
                labels[label][key] = [pt['center'][0], pt['center'][1]]
    labels = {k: v for k, v in labels.items() if len(v) == 2}
    if len(labels) >= 2:
        print(f'Using {len(labels)} labeled points: {sorted(labels.keys())}')
        allpts = list(labels.values())
    else:
        print(f'Using {len(labels)} corresponding points')
        allpts = [pt for pt in allpts if len(pt) == 2]
    mat = []
    val2 = []
    for pts in allpts:
        pt1 = pts['1']
        pt2 = pts['2']
        val2.append(pt2[0])
        val2.append(pt2[1])
        if args.transform == 'AFFINE':
            mat.append([pt1[0], pt1[1], 1, 0, 0, 0])
            mat.append([0, 0, 0, pt1[0], pt1[1], 1])
        else:
            mat.append([pt1[0], pt1[1], 1, 0])
            mat.append([pt1[1], -pt1[0], 0, 1])
    proj = np.linalg.lstsq(mat, np.array([val2]).T, rcond=-1)[0].flatten().tolist()
    print(proj, mat, val2)
    if args.transform == 'AFFINE':
        return np.array([[proj[0], proj[1], proj[2]], [proj[3], proj[4], proj[5]], [0, 0, 1]])
    scale = (proj[0] ** 2 + proj[1] ** 2) ** 0.5
    proj[0] /= scale
    proj[1] /= scale
    return np.array([[proj[0], proj[1], proj[2]], [-proj[1], proj[0], proj[3]], [0, 0, 1]])


def main(args):
    print('\n>> CLI Parameters ...\n')
    pprint.pprint(vars(args))
    if not args.style1 or args.style1.startswith('{#control'):
        args.style1 = None
    if not args.style2 or args.style2.startswith('{#control'):
        args.style2 = None
    with ProgressHelper('Register Images', 'Registering images', args.progress) as prog:
        prog.message('Opening first image')
        ts1 = large_image.open(args.image1, style=args.style1)
        print('Image 1:')
        pprint.pprint(ts1.metadata)
        prog.message('Opening second image')
        ts2 = large_image.open(args.image2, style=args.style2)
        print('Image 2:')
        pprint.pprint(ts2.metadata)
        maxRes = max(ts1.sizeX, ts1.sizeY, ts2.sizeX, ts2.sizeY)
        reduce = 1
        while math.ceil(maxRes / reduce) >= args.maxResolution:
            reduce *= 2
        print(f'Using reduction factor of {reduce}' if reduce > 1 else
              'Using images at original size')
        sizeX = int(math.ceil(max(ts1.sizeX, ts2.sizeX) / reduce))
        sizeY = int(math.ceil(max(ts1.sizeY, ts2.sizeY) / reduce))
        print(f'Registration size {sizeX} x {sizeY}')

        prog.message('Fetching first image')
        img1 = get_image(ts1, sizeX, sizeY, args.frame1, args.annotationID1, args, reduce)
        prog.message('Fetching second image')
        img2 = get_image(ts2, sizeX, sizeY, args.frame2, args.annotationID2, args, reduce)
        if isinstance(img1, list) or isinstance(img2, list):
            full = register_points(args, img1, img2)
        else:
            prog.message('Registering')
            sr = pystackreg.StackReg(getattr(
                pystackreg.StackReg, args.transform, pystackreg.StackReg.AFFINE))
            sr.register_transform(img1, img2)
            prog.message('Registered')
            print('Direct result')
            print(sr.get_matrix())
            full = sr.get_matrix().copy()
            full[0][2] *= reduce
            full[1][2] *= reduce
        print('Full result')
        print(full)
        print('Inverse')
        inv = np.linalg.inv(full)
        print(inv)
        print('Transforming image')
        prog.message('Transforming')
        transform_images(ts1, ts2, inv, args.outputSecondImage, args.outputMergedImage)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
