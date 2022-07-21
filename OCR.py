import LineExtraction as le
import CharacterRecognition as cr
import CropAndNormalize as cn
from PIL import Image as im

def main():
    ############
    # PARAMETERS
    ############
    learning_Rate = 0.5
    momentum = 1
    target_Error = 0.001
    number_of_hidden_neurons = 80
    width = 18
    height = 16
    number_of_training_samples = 4
    document_location = 'paragraphs/ocr1.png'

    # prepare image
    print('Loading Image...')
    input_image = im.open('%s' % document_location)  # open image
    input_image_black_and_white = cn.convert_to_black_and_white(input_image)  # black = 0, white = 1
    input_image_black_and_white = cn.toggle_ones_and_zeros(input_image_black_and_white)  # black = 1, white = 0
    input_image.show()
    print('Image Loaded')

    # Initialize and train network
    print('Training In Progress...')
    Wi_h, Wh_o, Bh, Bo = cr.initialize_weights(width, height, number_of_hidden_neurons)  # initialize weights
    Wi_h, Wh_o, Bh, Bo = cr.train_net(Wi_h, Wh_o, Bh, Bo, height, width, number_of_training_samples, learning_Rate, momentum, target_Error)  # train net
    print('Neural Net Trained')

    # recognize paragraphs, lines, words, and characters in image
    [cropped_lines_list, number_of_lines, top_of_lines, bottom_of_lines] = le.crop_lines(
        input_image_black_and_white)  # Extract Lines
    location_of_new_lines = le.crop_paragraphs(number_of_lines, top_of_lines, bottom_of_lines)  # get location of new lines
    number_of_new_lines = len(location_of_new_lines)
    lines_contents = []

    # Loop for all Lines
    for line in range(0, number_of_lines):
        [cropped_characters_list, number_of_characters, left_of_characters, right_of_characters] = le.cropCharacters(
            cropped_lines_list[line], number_of_lines)  # Characters from Line
        recognized_character_list = []
        # Loop for all characters in lines
        for character in range(0, number_of_characters):
            input_cropped_BW = cn.crop(cropped_characters_list[character])  # crop image to get the character only
            input_normalized = cn.normalize(input_cropped_BW, width, height)  # normalize image to fit neural network input size

            output = cr.recognize_character(input_normalized, Wi_h, Wh_o, Bh, Bo)  # normalized image is sent to recognition
            recognized_character_list.append(output)  # save characters found in line

            print('character number %d is %s' % (character, output))

        # crop Words from Line
        [words, number_of_words] = le.crop_words(recognized_character_list, left_of_characters, right_of_characters)  # form words from characters found in line
        lines_contents.append(words)  # save words

    print(lines_contents)

    # write output to file
    file_name = open('OCR_OUTPUT.txt', 'w')
    new_lines_index = 0
    for line in range(0, len(lines_contents)):
        if line > 0:  # No NewLine if its the first line
            file_name.write('\n')
        for word in range(0, len(lines_contents[line])):  # writing words
            file_name.write(lines_contents[line][word])
            file_name.write(' ')

        if line == location_of_new_lines[new_lines_index]:  # paragraphs spacing
            file_name.write('\n')
            if number_of_new_lines - 1 > new_lines_index:
                new_lines_index += 1
    file_name.close()


if __name__ == "__main__":
    main()