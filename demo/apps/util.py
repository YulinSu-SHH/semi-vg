def parallel_call(*fuc):
    def inner_call(image, instruction):
        out_imgs = []
        for ifunc in fuc:
            res, ood = ifunc(image, instruction)
            out_imgs.append(res)
        return out_imgs

    return inner_call
