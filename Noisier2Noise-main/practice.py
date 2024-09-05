from PIL import Image

def crop_image(image_path, output_path, crop_box):
    # Open the image
    image = Image.open(image_path)

    # Crop the image using the specified box
    cropped_image = image.crop(crop_box)

    # Save the cropped image
    cropped_image.save(output_path)

# Define the paths for the two images and their respective output paths
image1_path = r".\results\Set30\imgs\exp7\1th_overlap_mean.png"
output1_path = r".\results\Set30\imgs\exp7\1th_overlap_mean_cropped.png"

image2_path = r".\results\Set30\imgs\exp7\1th_prediction.png"
output2_path = r".\results\Set30\imgs\exp7\1th_prediction_cropped.png"

# = r".\results\Set22\imgs\exp9\1th_clean.png"
#output3_path = r".\results\Set22\imgs\exp9\1th_clean_cropped2.png"

# Define the crop box (left, upper, right, lower)
x_start =130
y_start = 90
crop_box = (x_start, y_start, x_start + 75, y_start + 75)  # 75x75 pixels

# Crop the first image
crop_image(image1_path, output1_path, crop_box)
crop_image(image2_path, output2_path, crop_box)
#crop_image(image3_path, output3_path, crop_box)
