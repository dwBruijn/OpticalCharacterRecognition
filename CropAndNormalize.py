import numpy as np
from PIL import Image as im


"""
    Detect black pixels in an image and remove the white pixels around them
"""


def convert_to_black_and_white(image):
    black_and_white = image.convert('1')
    black_and_white = np.array(black_and_white) * 1

    return black_and_white


def toggle_ones_and_zeros(black_and_white):
    return (black_and_white ^ 1)


def crop(black_and_white_toggled):
    [number_of_row_pixels, number_of_column_pixels] = black_and_white_toggled.shape  # find array dimensions

    # finding the left and right side
    vertical_sum_of_black_pixels = np.sum(black_and_white_toggled, axis=0)  # gives a list of number of black pixels in each column
    left_detected = False
    for i in range(0, number_of_column_pixels):
        if vertical_sum_of_black_pixels[i] > 0 and left_detected == False:  # there is a black pixel in this column
            left_detected = True
            left = i  # left
        elif vertical_sum_of_black_pixels[i] > 0 and left_detected == True:
            right = i  # right

    # finding the top and bottom side
    horizontal_sum_of_black_pixels = np.sum(black_and_white_toggled, axis=1)  # gives a list of number of black pixels in each row
    top_detected = False
    for i in range(0, number_of_row_pixels):
        if horizontal_sum_of_black_pixels[i] > 0 and top_detected == False:  # there is a black pixel in this column
            top_detected = True
            top = i  # top
        elif horizontal_sum_of_black_pixels[i] > 0 and top_detected == True:
            bottom = i  # bottom

    v_cropped_black_and_white_array = black_and_white_toggled[:, (range(left, right + 1))]
    final_cropped_black_and_white = v_cropped_black_and_white_array[(range(top, bottom + 1)), :]
    # Transform array back to image
    return final_cropped_black_and_white


def normalize(character_in, width, height):
    # using Hamming normalization
    character_in = im.fromarray((character_in * 255).astype(np.uint8))
    normalized = character_in.resize((width, height), im.HAMMING)  # normalize image to desired dimensions
    normalized_array = np.array(normalized)
    return normalized_array


################
# TESTING MODULE
################
def main():
    input_image_filename = "samples/a0.png"
    input_image = im.open(input_image_filename)  # load input image
    BW = convert_to_black_and_white(input_image)  # convert to black and white
    toggled = toggle_ones_and_zeros(BW)  # make 1-> black, 0-> white
    input_cropped_black_and_white = crop(toggled)  # crop image
    input_cropped_black_and_white_toggled = toggle_ones_and_zeros(input_cropped_black_and_white)
    img = im.fromarray((input_cropped_black_and_white_toggled * 255).astype(np.uint8))  # convert array back to image
    input_image.show()
    img.show()


if __name__ == "__main__":
    main()

