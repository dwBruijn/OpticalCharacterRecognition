from PIL import Image as im
import numpy as np
import CropAndNormalize as cn


# crop paragraphs from image
def crop_paragraphs(number_of_lines, top_of_lines, bottom_of_lines):
    location_of_new_lines = []
    for i in range(1, number_of_lines):
        whiteSpaceDistance = top_of_lines[i] - bottom_of_lines[i-1]
        if whiteSpaceDistance > 60:  # > 60 white pixels => new paragraph
            location_of_new_lines.append(i-1)

    return location_of_new_lines


""" 
crop lines from paragraph
returns the list of pixel values between top_of_line and bottom_of_line for all lines in paragraph
Example
000000000000000   -> white space
101010000000011   -> top of line
101010101000110   -> line content
101000101111010   -> bottom of line
000000000000000   -> white space
"""
def crop_lines(black_and_white):
    [number_row_pixels, number_column_pixels] = black_and_white.shape
    h_first_black_pixel_detected = False
    first_black_pixel_row = 0
    last_black_pixel_row = 0
    top_of_lines = []
    bottom_of_lines = []

    for i in range(0, number_row_pixels):
        sum_of_all_pixels_in_row_i = sum(black_and_white[i, :])
        if sum_of_all_pixels_in_row_i >= 1 and h_first_black_pixel_detected == False:  # detects first horizontal row
            h_first_black_pixel_detected = True
            first_black_pixel_row = i
            last_black_pixel_row = i
        elif sum_of_all_pixels_in_row_i >= 1 and h_first_black_pixel_detected == True:  # detects last horizontal row
            last_black_pixel_row = i
        elif sum_of_all_pixels_in_row_i < 1 and h_first_black_pixel_detected == True:  # detects a white row
            h_first_black_pixel_detected = False
            top_of_lines.append(first_black_pixel_row)  # save first_black_pixels in a list
            bottom_of_lines.append(last_black_pixel_row)  # save last_black_pixels in a list

    # creating a list that contains all cropped Lines
    number_of_lines = len(top_of_lines)
    croppedLinesList = []
    for i in range(0, number_of_lines):  # make a list containing croppedLines
        croppedLine = black_and_white[(range(top_of_lines[i], bottom_of_lines[i])), :]
        croppedLinesList.append(croppedLine)

    return croppedLinesList, number_of_lines, top_of_lines, bottom_of_lines


# crop characters from line
def cropCharacters(cropped_line, number_of_lines):
    [number_of_line_row_pixels, number_of_line_column_pixels] = cropped_line.shape
    left_detected = False
    first_black_pixel_column = 0
    last_black_pixel_column = 0
    left_of_characters = []
    right_of_characters = []

    for i in range(0, number_of_line_column_pixels):
        sum_of_all_pixels_in_coloumn_i = sum(cropped_line[:, i])
        if sum_of_all_pixels_in_coloumn_i >= 1 and left_detected == False:  # left found
            left_detected = True
            first_black_pixel_column = i
            last_black_pixel_column = i
        elif sum_of_all_pixels_in_coloumn_i >= 1 and left_detected == True:  # detects last horizontal row
            last_black_pixel_column = i
        elif sum_of_all_pixels_in_coloumn_i < 1 and left_detected == True:  # detects a white row
            left_detected = False
            left_of_characters.append(first_black_pixel_column)  # save left sides of characters in a list
            right_of_characters.append(last_black_pixel_column)  # save right sides of characters in a list

    # creating a list that contains all cropped Lines
    numberOfCharacters = len(left_of_characters)
    cropped_characters_list = []
    for i in range(0, numberOfCharacters):  # make a list containing cropped lines
        cropped_character = cropped_line[:, (range(left_of_characters[i], right_of_characters[i]))]
        cropped_characters_list.append(cropped_character)

    return cropped_characters_list, numberOfCharacters, left_of_characters, right_of_characters


# crop words from line
def crop_words(recognized_characters_list, left_of_characters, right_of_characters):
    number_of_characters = len (left_of_characters)
    location_of_spaces = [0]
    words = []
    for i in range(1, number_of_characters):
        white_space_between_characters = left_of_characters[i] - right_of_characters[i-1]
        if white_space_between_characters > 20 :
            location_of_spaces.append(i)
        if i == number_of_characters-1:
            location_of_spaces.append(number_of_characters)

    for i in range(0,len(location_of_spaces)-1):
        first_character_in_word = location_of_spaces[i]
        last_character_in_word = location_of_spaces[i+1]
        word = recognized_characters_list[first_character_in_word:last_character_in_word]
        word = ''.join(word)
        words.append(word)
    number_of_words = len(words)

    return words, number_of_words
