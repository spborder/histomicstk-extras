import copy
import logging
import math
import os
import pprint
import sys
import tempfile
import uuid
import json

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
import skimage.measure
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

def make_annotation_from_shape(shape_list,name,properties)->dict:
    """
    Take a Shapely shape object (MultiPolygon or Polygon or GeometryCollection) and return the corresponding annotation 
    """
    annotation_dict = {
        "annotation": {
            "name": name,
            "elements": []
        }
    }
    for shape in shape_list:
        
        if shape.geom_type=='Polygon' and shape.is_valid:
            # Exterior "shell" coordinates
            coords = list(shape.exterior.coords)
            """
            # Ignoring holes for tissue
            hole_coords = list(shape.interiors)
            hole_list = []
            for h in hole_coords:
                hole_list.append([
                    [i[0],i[1]]
                    for i in list(h.coords)
                ])
            """
            hole_list = []
            annotation_dict['annotation']['elements'].append({
                'type': 'polyline',
                'points': [list(i)+[0] for i in coords],
                'holes': hole_list,
                'id': uuid.uuid4().hex[:24],
                'closed': True,
                'user': properties
            })

    return annotation_dict

def get_tissue_mask(img:np.array,brightfield:bool)->np.array:
    """
    Getting mask of tissue to constrain annotations

    """
    img = np.squeeze(np.mean(img,axis=-1))

    if brightfield:
        img = 255-img

    threshold_val = skimage.filters.threshold_otsu(img)
    tissue_mask = img <= threshold_val

    tissue_mask = skimage.morphology.remove_small_holes(tissue_mask,area_threshold = 150)

    return tissue_mask

def create_annotation(img:np.array, scale_factor:int)->dict:
    """
    Converting mask to large_image annotations
    """

    labeled_mask = skimage.measure.label(img>0)
    pieces = np.unique(labeled_mask).tolist()
    tissue_shape_list = []
    for piece in pieces[1:]:
        piece_contours = skimage.measure.find_contours(labeled_mask==piece)

        for contour in piece_contours:

            poly_list = [(i[1]*scale_factor,i[0]*scale_factor) for i in contour]
            if len(poly_list)>2:
                obj_polygon = shapely.geometry.Polygon(poly_list)

                if not obj_polygon.is_valid:
                    made_valid = shapely.validation.make_valid(obj_polygon)

                    if made_valid.geom_type=='Polygon':
                        tissue_shape_list.append(made_valid)
                    elif made_valid.geom_type in ['MultiPolygon','GeometryCollection']:
                        for g in made_valid.geoms:
                            if g.geom_type=='Polygon':
                                tissue_shape_list.append(g)
                else:
                    tissue_shape_list.append(obj_polygon)

    merged_mask = shapely.ops.unary_union(tissue_shape_list)
    if merged_mask.geom_type=='Polygon':
        merged_mask = [merged_mask]
    elif merged_mask.geom_type in ['MultiPolygon','GeometryCollection']:
        merged_mask = merged_mask.geoms

    annotation = make_annotation_from_shape(merged_mask,'Registration Intermediate',{})

    return annotation




def get_image(ts, sizeX, sizeY, frame, annotID, args, reduce, brightfield,save_intermediate):
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
    
    regionparams['output'] = dict(maxWidth=ts.sizeX // reduce, maxHeight=ts.sizeY // reduce)
    img = ts.getRegion(**regionparams)[0]
    tissue_mask = get_tissue_mask(img,brightfield)

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

    img *= np.uint8(tissue_mask)

    if save_intermediate:
        tissue_mask_annotation = create_annotation(tissue_mask)
        img_annotation = create_annotation(img)

        return img, [tissue_mask_annotation,img_annotation]
    else:
        return img, []


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
        
        base_plugin_dir = '/mnt/girder_worker/'+os.listdir('/mnt/girder_worker')[0]+'/'
        # Downloading image items to this location

        gc = girder_client.GirderClient(apiUrl = args.girderApiUrl)
        gc.setToken(args.girderToken)

        image_1_name = gc.get(f'/file/{args.image1}')['name']
        gc.downloadFile(args.image1,base_plugin_dir+image_1_name)

        image_2_name = gc.get(f'/file/{args.image2}')['name']
        gc.downloadFile(args.image2,base_plugin_dir+image_2_name)

        image_1_path = base_plugin_dir+image_1_name
        image_2_path = base_plugin_dir+image_2_name
                
        prog.message('Opening first image')
        ts1 = large_image.open(image_1_path, style=args.style1)
        print('Image 1:')
        pprint.pprint(ts1.metadata)
        prog.message('Opening second image')
        ts2 = large_image.open(image_2_path, style=args.style2)
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
        img1, intermediates1 = get_image(ts1, sizeX, sizeY, args.frame1, args.annotationID1, args, reduce,args.image1_brightfield,args.save_intermediate_annotations)
        prog.message('Fetching second image')
        img2, intermediates2 = get_image(ts2, sizeX, sizeY, args.frame2, args.annotationID2, args, reduce,args.image2_brightfield,args.save_intermediate_annotations)
        
        # Posting intermediate        
        if args.save_intermediate_annotations:

            _ = gc.post(f'/annotation/item/{args.image1}',
                             data = json.dumps(intermediates1),
                             headers = {
                                 'X-HTTP-Method': 'POST',
                                 'Content-Type':'application/json'
                             })
            
            _ = gc.post(f'/annotation/item/{args.image2}',
                        data = json.dumps(intermediates2),
                        headers = {
                            'X-HTTP-Method':'POST',
                            'Content-Type':'application/json'
                        })

        
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
