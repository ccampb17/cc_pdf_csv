# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Requires Python 3.6 or higher due to f-strings

# Import libraries
import platform
from tempfile import TemporaryDirectory
from pathlib import Path

import PIL.Image
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy
import pytesseract
import numpy as np
import pandas as pd
import os

if platform.system() == "Windows":
    # We may need to do some additional downloading and setup...
    # Windows needs a PyTesseract Download
    # https://github.com/UB-Mannheim/tesseract/wiki/Downloading-Tesseract-OCR-Engine

    pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\c-cam\AppData\Local\Tesseract-OCR\tesseract.exe")

    # Windows also needs poppler_exe
    path_to_poppler_exe = Path(r"C:\Program Files (x86)\poppler-22.11.0\Library\bin")

    # Put our output files in a sane place...
    out_directory = Path(r"~\Desktop").expanduser()
else:
    out_directory = Path("~").expanduser()

# Path of the Input pdf
PDF_file = Path(r"C:\Users\c-cam\GTA data team Dropbox\Bastiat\0 projects\044 PDF to CSV\Extraordinary_Gazettes_2306-15-Department_of_Trade_and_Investment_Policies.pdf")

# Store all the pages of the PDF in a variable
image_file_list = []

text_file = out_directory / Path("out_text.txt")

def main():
    ''' Main execution point of the program'''
    with TemporaryDirectory() as tempdir:
        # Create a temporary directory to hold our temporary images.

        """
        Part #1 : Converting PDF to images
        """
        print(f'Converting {PDF_file} to images...')
        if platform.system() == "Windows":
            pdf_pages = convert_from_path(
            PDF_file, 500, poppler_path=path_to_poppler_exe
            )
        else:
            pdf_pages = convert_from_path(PDF_file, 1000)
            # Read in the PDF file at 500 DPI

        # Iterate through all the pages stored above
        for page_enumeration, page in enumerate(pdf_pages, start=4):
            # enumerate() "counts" the pages for us.

            # Create a file name to store the image
            filename = f"{tempdir}\page_{page_enumeration:03}.jpg"
            print(f'Converting {page_enumeration}...')

            # Declaring filename for each page of PDF as JPG
            # For each page, filename will be:
            # PDF page 1 -> page_001.jpg
            # PDF page 2 -> page_002.jpg
            # PDF page 3 -> page_003.jpg
            # ....
            # PDF page n -> page_00n.jpg

            # Save the image of the page in system
            page.save(filename, "JPEG")
            image_file_list.append(filename)
            print(f'Done')

        """
        Part #2 - Recognizing text from the images using OCR
        """
        df_result = []
        with open(text_file, "a") as output_file:
            # Open the file in append mode so that
            # All contents of all images are added to the same file
            i = 0
            # Iterate from 1 to total number of pages
            for image_file in image_file_list:
                i += 1  # for filenames
                # Set filename to recognize text from
                # Again, these files will be:
                # page_1.jpg
                # page_2.jpg
                # ....
                # page_n.jpg
                print(f'Tesseracting {image_file}...')
                image = np.array(PIL.Image.open(image_file))
                result = image.copy()
                # Recognize the text as string in image using pytesserct

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                # Remove horizontal lines
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
                remove_horizontal = cv2.morphologyEx(thresh,
                                                     cv2.MORPH_OPEN,
                                                     horizontal_kernel,
                                                     iterations=2)
                cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                for c in cnts:
                    cv2.drawContours(result, [c], -1, (255, 255, 255), 5)

                # Remove vertical lines
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
                remove_vertical = cv2.morphologyEx(thresh,
                                                   cv2.MORPH_OPEN,
                                                   vertical_kernel,
                                                   iterations=2)
                cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                for c in cnts:
                    cv2.drawContours(result, [c], -1, (255, 255, 255), 5)


                tessdata_dir_config = r'--tessdata-dir "C:\Users\c-cam\AppData\Local\Tesseract-OCR\tessdata"'
                tessdata_dir_config += ' --oem 3 --psm 1'

                text = pytesseract.image_to_string(np.array(result),
                                                 lang='eng',
                                                 config=tessdata_dir_config,
                                                 output_type='string')

                #df_pd = pd.DataFrame(df)
                # out_fname = f'output_csv_{i}.csv'
                # df_pd.to_csv(path_or_buf=f'{os.path.join(os.getcwd(), out_fname)}')

                #print(f'made a df of {image_file}, now doing it to the text file')

                # df_result.append(df)
                # text = str(((pytesseract.image_to_string(Image.open(image_file)))))

                # string preprocessing if required

                text = text.replace("-\n", "")

                print('Writing to file...')

                output_file.write(text)

    print('finished!')

if __name__ == "__main__":

    main()