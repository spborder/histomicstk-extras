import os
import pprint
import shutil
import stat

import large_image
from histomicstk.cli.utils import CLIArgumentParser


def main(args):
    print('\n>> CLI Parameters ...\n')
    pprint.pprint(vars(args))
    if not args.style or args.style.startswith('{#control'):
        args.style = None
    if not args.frame or args.frame.startswith('{#control'):
        args.frame = None
    elif not args.frame.isdigit():
        msg = 'The given frame value is not an integer'
        raise Exception(msg)
    else:
        args.frame = int(args.frame)
    ts = large_image.open(args.image, style=args.style)
    region = None
    if any(val != -1 for val in args.roi):
        region = {
            'left': args.roi[0],
            'top': args.roi[1],
            'width': args.roi[2],
            'height': args.roi[3],
        }
    result, _ = ts.getRegion(encoding='TILED', frame=args.frame, region=region)
    print(result, os.path.getsize(result))
    shutil.copy(result, args.outputImage)
    os.chmod(args.outputImage,
             os.stat(args.outputImage).st_mode | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
