import os


def get_full_img_dict(images, sep):
    """Returns a dict with keys the full images names and values the lst of corresponding images.
    sep is the string which separates the full img names
    Assumes all img parts have same class (located in same dir) which will be attributed to full image"""
    full_images_dict = {}
    for fpath in images:
        cls = os.path.basename(os.path.dirname(fpath))
        file, ext = os.path.splitext(os.path.basename(fpath))
        fi = os.path.join(cls, f'{file.split(sep)[0] if sep in file else file}{ext}')
        if fi in full_images_dict:
            full_images_dict[fi].append(fpath)
        else:
            full_images_dict[fi] = [fpath]
    return full_images_dict


def crop_im(im, bbox):
    wmin, hmin, wmax, hmax = bbox
    return im[hmin:hmax+1, wmin:wmax+1]

