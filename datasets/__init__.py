from .hybrid_dataloader import build_hybrid

def build_dataset(image_set, args):
    if args.dataset_file == 'hybrid':
        return build_hybrid(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
