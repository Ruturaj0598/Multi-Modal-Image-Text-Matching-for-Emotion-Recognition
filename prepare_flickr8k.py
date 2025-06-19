import argparse 
import csv 
import cv2
import numpy as np 
import os
import pandas as pd
from collections import Counter
import pickle
import string

class flickr_caption:
    def __init__(self, filename, caption):
        self.filename = filename
        self.caption = caption
        self.im_size = []
        self.tokenized_caption = []
        self.caption_indices = []
    
    def set_imsize(self, image_size):
        """Set image size from loaded image"""
        self.im_size = [image_size[0], image_size[1]]  # [height, width]
    
    def tokenize_caption(self, caption, vocab=None, max_length=50):
        """Tokenize caption and convert to indices"""
        # Simple tokenization - split by spaces and remove punctuation
        caption = caption.lower()
        caption = caption.translate(str.maketrans('', '', string.punctuation))
        tokens = caption.split()
        
        # Add special tokens
        tokens = ['<start>'] + tokens + ['<end>']
        
        if vocab is not None:
            # Convert to indices
            indices = []
            for token in tokens:
                if token in vocab:
                    indices.append(vocab[token])
                else:
                    indices.append(vocab['<unk>'])  # Unknown token
            
            # Pad or truncate to max_length
            if len(indices) < max_length:
                indices.extend([vocab['<pad>']] * (max_length - len(indices)))
            else:
                indices = indices[:max_length]
            
            self.caption_indices = indices
        
        self.tokenized_caption = tokens

def build_vocabulary(captions, min_freq=2):
    """Build vocabulary from captions"""
    # Tokenize all captions and count word frequencies
    all_tokens = []
    for caption in captions:
        caption = caption.lower()
        caption = caption.translate(str.maketrans('', '', string.punctuation))
        tokens = caption.split()
        all_tokens.extend(tokens)
    
    # Count frequencies
    word_counts = Counter(all_tokens)
    
    # Create vocabulary with special tokens
    vocab = {
        '<pad>': 0,
        '<start>': 1, 
        '<end>': 2,
        '<unk>': 3
    }
    
    # Add words that meet minimum frequency
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

def prepare_flickr_data(captions_file, images_dir, save_dir, dataset_type='train', 
                       generate_npy=False, debug_mode=False, vocab=None, max_caption_length=50):
    """
    Prepare Flickr8k data similar to Emotic preprocessing
    """
    data_set = list()
    
    if generate_npy:
        context_arr = list()  # Full images (224x224)
        body_arr = list()     # Same as context for Flickr8k (128x128) 
        cat_arr = list()      # Caption indices (equivalent to categorical emotions)
        cont_arr = list()     # Caption lengths (equivalent to continuous values)
    
    # Read captions file
    df = pd.read_csv(captions_file)
    
    to_break = 0
    path_not_exist = 0
    caption_too_short = 0
    idx = 0
    
    print(f"Processing {len(df)} caption entries for {dataset_type}...")
    
    for row_idx, row in df.iterrows():
        filename = row['image']
        caption = row['caption']
        
        # Create flickr_caption object
        fc = flickr_caption(filename, caption)
        
        try:
            image_path = os.path.join(images_dir, filename)
            if not os.path.exists(image_path):
                path_not_exist += 1
                if debug_mode:
                    print('Path not existing:', image_path)
                continue
            else:
                # Load and resize image
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                fc.set_imsize(image.shape)
                
                # Resize images to match Emotic format
                context_cv = cv2.resize(image, (224, 224))  # Context image
                body_cv = cv2.resize(image, (128, 128))     # Body image (same as context)
                
        except Exception as e:
            to_break += 1
            if debug_mode:
                print(f'Breaking at idx={row_idx}, {idx} due to exception={e}')
            continue
        
        # Tokenize caption
        fc.tokenize_caption(caption, vocab, max_caption_length)
        
        # Skip if caption is too short (after tokenization)
        meaningful_tokens = [t for t in fc.tokenized_caption if t not in ['<start>', '<end>', '<pad>']]
        if len(meaningful_tokens) < 3:
            caption_too_short += 1
            continue
        
        data_set.append(fc)
        
        if generate_npy:
            context_arr.append(context_cv)
            body_arr.append(body_cv)
            cat_arr.append(np.array(fc.caption_indices))  # Caption as categorical data
            cont_arr.append(np.array([len(meaningful_tokens)]))  # Caption length as continuous
        
        if idx % 1000 == 0 and not debug_mode:
            print(f" Preprocessing data. Index = {idx}")
        elif idx % 20 == 0 and debug_mode:
            print(f" Preprocessing data. Index = {idx}")
        
        idx += 1
        
        # Debug mode early exit
        if debug_mode and idx >= 100:
            print(f' ######## Breaking data prep step {idx}, {row_idx} ######')
            print(f'Stats: to_break={to_break}, path_not_exist={path_not_exist}, caption_too_short={caption_too_short}')
            if context_arr:
                cv2.imwrite(os.path.join(save_dir, 'context_sample.png'), context_arr[-1])
                cv2.imwrite(os.path.join(save_dir, 'body_sample.png'), body_arr[-1])
            break
    
    print(f'Final stats: to_break={to_break}, path_not_exist={path_not_exist}, caption_too_short={caption_too_short}')
    
    # Save CSV file
    csv_path = os.path.join(save_dir, f"{dataset_type}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', dialect='excel')
        row = ['Index', 'Filename', 'Image_Size', 'Caption', 'Tokenized_Caption', 'Caption_Indices']
        filewriter.writerow(row)
        for idx, fc in enumerate(data_set):
            row = [idx, fc.filename, fc.im_size, fc.caption, fc.tokenized_caption, fc.caption_indices]
            filewriter.writerow(row)
    print(f'Wrote file {csv_path}')
    
    # Save npy files (same format as Emotic)
    if generate_npy:
        context_arr = np.array(context_arr)
        body_arr = np.array(body_arr)
        cat_arr = np.array(cat_arr)
        cont_arr = np.array(cont_arr)
        
        print(f'Array shapes: context={context_arr.shape}, body={body_arr.shape}, cat={cat_arr.shape}, cont={cont_arr.shape}')
        
        np.save(os.path.join(save_dir, f'{dataset_type}_context_arr.npy'), context_arr)
        np.save(os.path.join(save_dir, f'{dataset_type}_body_arr.npy'), body_arr)
        np.save(os.path.join(save_dir, f'{dataset_type}_cat_arr.npy'), cat_arr)
        np.save(os.path.join(save_dir, f'{dataset_type}_cont_arr.npy'), cont_arr)
        
        print(f'Saved npy arrays with shapes: {context_arr.shape}, {body_arr.shape}, {cat_arr.shape}, {cont_arr.shape}')
    
    print(f'Completed generating {dataset_type} data files')
    return data_set

def split_flickr_data(captions_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split Flickr8k data into train/val/test sets by images"""
    df = pd.read_csv(captions_file)
    
    # Get unique images and shuffle
    unique_images = df['image'].unique()
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_images)
    
    n_total = len(unique_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_images = unique_images[:n_train]
    val_images = unique_images[n_train:n_train + n_val]
    test_images = unique_images[n_train + n_val:]
    
    # Create split dataframes
    train_df = df[df['image'].isin(train_images)]
    val_df = df[df['image'].isin(val_images)]
    test_df = df[df['image'].isin(test_images)]
    
    print(f'Data split: Train={len(train_df)} captions ({len(train_images)} images), '
          f'Val={len(val_df)} captions ({len(val_images)} images), '
          f'Test={len(test_df)} captions ({len(test_images)} images)')
    
    return train_df, val_df, test_df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Flickr8k data directory')
    parser.add_argument('--captions_file', type=str, default='captions.txt', help='Caption file name')
    parser.add_argument('--images_dir', type=str, default='Images', help='Images directory name')
    parser.add_argument('--save_dir_name', type=str, default='flickr8k_pre', help='Directory name to store preprocessed data')
    parser.add_argument('--label', type=str, default='all', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--generate_npy', action='store_true', help='Generate npy files')
    parser.add_argument('--debug_mode', action='store_true', help='Debug mode')
    parser.add_argument('--max_caption_length', type=int, default=50, help='Maximum caption length')
    parser.add_argument('--min_word_freq', type=int, default=2, help='Minimum word frequency for vocabulary')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    captions_path = os.path.join(args.data_dir, args.captions_file)
    images_path = os.path.join(args.data_dir, args.images_dir)
    save_path = os.path.join(args.data_dir, args.save_dir_name)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print('Loading and splitting captions...')
    
    # Split data into train/val/test
    train_df, val_df, test_df = split_flickr_data(captions_path)
    
    # Build vocabulary from training captions only
    print('Building vocabulary...')
    vocab = build_vocabulary(train_df['caption'].tolist(), args.min_word_freq)
    
    # Save vocabulary
    vocab_path = os.path.join(save_path, 'vocabulary.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    # Also save as CSV for easy inspection
    vocab_csv_path = os.path.join(save_path, 'vocabulary.csv')
    with open(vocab_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'index'])
        for word, idx in vocab.items():
            writer.writerow([word, idx])
    
    print(f'Vocabulary size: {len(vocab)}')
    
    # Process datasets
    if args.label.lower() == 'all':
        datasets = [
            ('train', train_df),
            ('val', val_df), 
            ('test', test_df)
        ]
    else:
        if args.label.lower() == 'train':
            datasets = [('train', train_df)]
        elif args.label.lower() == 'val':
            datasets = [('val', val_df)]
        else:
            datasets = [('test', test_df)]
    
    for dataset_type, df in datasets:
        print(f'\nProcessing {dataset_type} dataset...')
        
        # Save split caption file
        split_captions_path = os.path.join(save_path, f'{dataset_type}_captions.csv')
        df.to_csv(split_captions_path, index=False)
        
        # Prepare data
        prepare_flickr_data(
            split_captions_path,
            images_path,
            save_path,
            dataset_type=dataset_type,
            generate_npy=args.generate_npy,
            debug_mode=args.debug_mode,
            vocab=vocab,
            max_caption_length=args.max_caption_length
        )
    
    print(f'\nPreprocessing completed! Files saved in: {save_path}')
