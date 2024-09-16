import torch.utils.data as data  # For dataset management
from PIL import Image  # To handle images
import os  # For OS operations like path handling
import os.path  # Additional path operations
import json  # For handling JSON data
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # For plotting
import text_util  # Custom text utilities (presumed external file)
import torch  # PyTorch for deep learning functionality
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader  # To handle video reading
import copy  # For copying objects
import torch.utils.data.sampler as sampler  # For creating custom samplers
from data import Dictionary  # Dictionary class for managing words

class Twitch(data.Dataset):
    def __init__(self, root, list_file, number=1000, transform=None, text_transform=None,
                 prod_Img=True, prod_Text=True, multi_frame=1, text_window=150, text_delay=0, 
                 gt_range=0.25, word=False, corpus=None):
        """
        Initializes the Twitch dataset class.
        
        Args:
        root (str): Root directory for the dataset.
        list_file (str): File path to the dataset list.
        number (int): Number of samples.
        transform (callable, optional): Image transformation pipeline.
        text_transform (callable, optional): Text transformation pipeline.
        prod_Img (bool): Flag to produce images.
        prod_Text (bool): Flag to produce text.
        multi_frame (int): Number of frames to load from videos.
        text_window (int): Window size for text loading.
        text_delay (int): Delay before starting text sequence.
        gt_range (float): Range for ground truth labels.
        word (bool): Flag to use words in dictionary.
        corpus (Dictionary): Custom dictionary for words.
        """
        self.root = root
        self.__load_set(list_file)  # Load video, text, and ground truth (GT) lists
        self.transform = transform
        self.text_transform = text_transform
        self.nums = number
        self.prod_Img = prod_Img
        self.prod_Text = prod_Text
        self.multi_frame = multi_frame  # Number of video frames to load per sample
        self.video_idx = 0
        self.gt_range = 1 - gt_range  # Ground truth range adjustment
        self.text_window = text_window  # How many lines of text to retrieve
        self.text_delay = text_delay  # How much delay before text starts
        self.WeightedSampling = []  # Initialize empty list for weighted sampling
        self.word = word
        self.corpus = corpus  # Dictionary for words (if needed)
        if self.word and corpus is None:
            self.__set_corpus()  # Set corpus if not provided and word flag is True

        # Fill in WeightedSampling list
        for gt in self.gt_list:
            self.WeightedSampling.extend(copy.copy(gt))

        sampling = np.array(self.WeightedSampling)  # Convert to NumPy array
        neg_idx = np.where(sampling == 0)[0]  # Indices of negative samples
        pos_idx = np.where(sampling == 1)[0]  # Indices of positive samples

        begin_pos = 0
        hl_frames = []  # High-likelihood frames for ground truth
        for it, cur_pos in enumerate(pos_idx):
            if it + 1 < len(pos_idx):
                if (pos_idx[it + 1] - cur_pos) > 1:  # If there is a gap
                    begin = int((it + 1 - begin_pos) * self.gt_range) + begin_pos
                    hl_frames.extend(pos_idx[begin:it])  # Append high-likelihood frames
                    begin_pos = it + 1

        sampling.fill(0)  # Reset sampling array to zero
        sampling[neg_idx] = len(sampling) / float(len(neg_idx))  # Set weights for negative samples
        sampling[hl_frames] = len(sampling) / float(len(hl_frames))  # Set weights for high-likelihood frames
        self.WeightedSampling = sampling
        
        # Cumulative sum for indexing video list
        self.sums = np.insert(np.cumsum([len(gt) for gt in self.gt_list]), 0, 0)
        print('Twitch Data Loader is ready.')

    def __load_set(self, set_file):
        """
        Loads the set file which contains paths to videos, text, and ground truth.
        
        Args:
        set_file (str): File path to the list of dataset files.
        """
        with open(set_file) as f:
            lines = f.readlines()  # Read all lines from set file

        video_list = []
        text_list = []
        gt_list = []
        
        for line in lines:
            line = line.strip('\n')  # Remove newline characters
            segs = line.split(' ')  # Split line by space (3 segments: video, text, gt)
            print('=> Load Video', segs)
            assert len(segs) == 3  # Ensure there are exactly 3 segments
            segs = [os.path.join(self.root, seg) for seg in segs]  # Convert to full paths

            video_list.append(segs[0])  # Append video path
            cap = FFMPEG_VideoReader(segs[0])  # Open video file
            cap.initialize()
            print(f'Video: frames({int(cap.nframes)})')  # Print frame count
            
            text = json.load(open(segs[1]))  # Load text JSON
            gt = np.load(open(segs[2]))  # Load ground truth numpy array
            print(f'GT: frames({len(gt)})')  # Print ground truth frame count
            
            text_list.append(text)  # Append text to list
            gt_list.append(gt)  # Append ground truth to list

        self.video_list = video_list
        self.text_list = text_list
        self.gt_list = gt_list

    def __set_corpus(self):
        """
        Sets the corpus by building a dictionary of words from text data.
        """
        pre_dict = Dictionary()  # Create a pre-processing dictionary
        for lines in self.text_list:
            for line in lines:
                if len(line) > 0:
                    words = line.split()  # Split lines into words
                    for word in words:
                        pre_dict.add_word(word)  # Add word to the pre-dictionary

        pro_dict = Dictionary()  # Create a final dictionary
        for key in pre_dict.count:  # Filter words by occurrence count
            if pre_dict.count[key] > 10:  # Only include words occurring more than 10 times
                pro_dict.add_word(key)
        self.corpus = pro_dict  # Set the corpus

    def __getitem__(self, index):
        """
        Retrieves the sample at the given index.
        
        Args:
        index (int): Sample index.
        
        Returns:
        tuple: (images, text, ground truth)
        """
        # Find the video first by matching index to cumulative sums
        vid = np.histogram(index, self.sums)
        assert np.sum(vid[0]) == 1  # Ensure exactly one video is selected
        vid = np.where(vid[0] > 0)[0][0]  # Get the video index

        vframe = index - self.sums[vid]  # Determine the frame index within the video
        imgs = []
        
        if self.prod_Img:  # If producing images
            cap = FFMPEG_VideoReader(self.video_list[vid])  # Open video
            cap.initialize()
            
            for i in range(self.multi_frame):  # Load multiple frames if specified
                if i == 0:
                    img = cap.get_frame(vframe / cap.fps)  # Get first frame
                else:
                    cap.skip_frames(n=9)  # Skip 9 frames
                    img = cap.read_frame()  # Get the next frame
                
                img = Image.fromarray(img)  # Convert to PIL image
                if self.transform:
                    img = self.transform(img)  # Apply image transformations
                imgs.append(img)

            imgs = [img.unsqueeze(0) for img in imgs]  # Add an extra dimension to images
            imgs = torch.cat(imgs, 0)  # Concatenate images along the batch dimension

        text = [] 
        if self.prod_Text:  # If producing text
            text = self.text_list[vid][min(vframe + self.text_delay, len(self.text_list[vid]))
                    : min(vframe + self.text_window + self.text_delay, len(self.text_list[vid]))]
            text = '\n'.join(text)  # Join text with newlines

        gt = self.gt_list[vid][vframe]  # Retrieve ground truth for the frame
        
        if len(text) == 0:
            text = ' '  # Ensure text is non-empty

        return imgs, text, gt  # Return images, text, and ground truth

    def __len__(self):
        """
        Returns the total number of samples for the dataset.
        """
        return len(self.WeightedSampling)  # Use weighted sampling length

class SampleSequentialSampler(sampler.Sampler):
    """
    Samples elements sequentially, always in the same order.
    
    Args:
    data_source (Dataset): Dataset to sample from.
    offset (int): Offset between the samples.
    """
    def __init__(self, data_source, offset):
        self.data_source = data_source
        self.offset = offset  # Offset determines sequence starting point

    def __iter__(self):
        """
        Provides an iterator for sequentially sampling the dataset.
        
        Returns:
        iterator: Sequential sampling iterator.
        """
        return iter([i + self.offset for i in range(len(self.data_source))])

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data_source)
