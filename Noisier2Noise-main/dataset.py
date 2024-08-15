from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import *


class ImageNetGray(Dataset):
    def __init__(self, data_dir='../all_datasets/ImageNet_1000_Gray/', noise='gauss_25', train=True, transform=None):
        super(ImageNetGray, self).__init__()

        self.noise_type, self.noise_intensity = noise.split('_')[0], float(noise.split('_')[1]) / 255.

        if train:
            self.clean_dir = os.path.join(data_dir, 'train')
        else:
            self.clean_dir = os.path.join(data_dir, 'test')

        self.clean_paths = sorted(make_dataset(self.clean_dir))

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        clean_path = self.clean_paths[index]
        clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE) / 255.
        if self.noise_type == 'gauss':
            noisy = clean + np.random.randn(*clean.shape) * self.noise_intensity
            noisier = noisy + np.random.randn(*clean.shape) * self.noise_intensity
        elif self.noise_type == 'poisson':
            noisy = np.random.poisson(clean * 255. * self.noise_intensity) / self.noise_intensity / 255.
            # Add Poisson
            noisier = noisy + (np.random.poisson(clean * 255. * self.noise_intensity) / self.noise_intensity / 255. - clean)
            # # Add Gaussian approximation
            # noisier = noisy + np.random.randn(*clean.shape) *
        else:
            raise NotImplementedError('wrong type of noise')
        clean, noisy, noisier = self.transform(clean), self.transform(noisy), self.transform(noisier)
        clean, noisy, noisier = clean.type(torch.FloatTensor), noisy.type(torch.FloatTensor), noisier.type(torch.FloatTensor)
        return {'clean': clean, 'noisy': noisy, 'noisier': noisier}

    def __len__(self):
        return len(self.clean_paths)


# from torch.utils.data import Dataset
# import torchvision.transforms as transforms
# from utils import *
#
# class ImageNetGray(Dataset):
#     def __init__(self, data_dir='../all_datasets/ImageNet_1000_Gray/', noise='gauss_25', train=True, transform=None, target_size=(256, 256), grayscale=True):
#         super(ImageNetGray, self).__init__()
#
#         self.noise_type, self.noise_intensity = noise.split('_')[0], float(noise.split('_')[1]) / 255.
#         self.target_size = target_size
#         self.grayscale = grayscale
#
#         if train:
#             self.clean_dir = os.path.join(data_dir, 'train')
#         else:
#             self.clean_dir = os.path.join(data_dir, 'test')
#
#         self.clean_paths = sorted(make_dataset(self.clean_dir))
#
#         if transform:
#             self.transform = transform
#         else:
#             self.transform = transforms.Compose([transforms.ToTensor()])
#
#     def __getitem__(self, index):
#         clean_path = self.clean_paths[index]
#         img = cv2.imread(clean_path)
#
#         # Convert to grayscale if specified
#         if self.grayscale:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         # Pad the image to the target size
#         clean = cv2.copyMakeBorder(
#             img,
#             0,  # Top padding
#             max(0, self.target_size[0] - img.shape[0]),  # Bottom padding
#             0,  # Left padding
#             max(0, self.target_size[1] - img.shape[1]),  # Right padding
#             cv2.BORDER_CONSTANT,  # Padding type
#             value=0  # Padding color (black)
#         )
#
#         if self.noise_type == 'gauss':
#             noisy = clean + np.random.randn(*clean.shape) * self.noise_intensity
#             noisier = noisy + np.random.randn(*clean.shape) * self.noise_intensity
#         elif self.noise_type == 'poisson':
#             noisy = np.random.poisson(clean * 255. * self.noise_intensity) / self.noise_intensity / 255.
#             noisier = noisy + (np.random.poisson(clean * 255. * self.noise_intensity) / self.noise_intensity / 255. - clean)
#         else:
#             raise NotImplementedError('Unsupported type of noise')
#
#         # Apply the transformation
#         if self.grayscale:
#             clean, noisy, noisier = self.transform(clean), self.transform(noisy), self.transform(noisier)
#         else:
#             clean = self.transform(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
#             noisy = self.transform(cv2.cvtColor(noisy.astype(np.uint8), cv2.COLOR_BGR2RGB))
#             noisier = self.transform(cv2.cvtColor(noisier.astype(np.uint8), cv2.COLOR_BGR2RGB))
#
#         clean, noisy, noisier = clean.type(torch.FloatTensor), noisy.type(torch.FloatTensor), noisier.type(torch.FloatTensor)
#         return {'clean': clean, 'noisy': noisy, 'noisier': noisier}
#
#     def __len__(self):
#         return len(self.clean_paths)
