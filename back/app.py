
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.morphology import skeletonize
from skimage.color import rgb2gray
from PIL import Image
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image, ImageOps
from io import BytesIO
from docx import Document
import pytesseract
import io
# pwd = os.getcwd()+"/back"
# os.chdir(pwd)


def decodeImage(data):
    # Gives us 1d array
    decoded = np.fromstring(data, dtype=np.uint8)
    # We have to convert it into (270, 480,3) in order to see as an image
    # decoded = decoded.reshape((decoded.shape))
    decoded = rgb2gray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
    return decoded


def pdf_to_cells(file):
    """
    Extracts the table cells of input image into a 2d array of image patches.
    Args:
        file (string):  Filepath to image file or pdf (only first page is
                        considered).
    Returns:
        result (numpy array):   Array of eqaul shape as input table, containing
                                table cells (numpy arrays). First and last row
                                are ignored.
    """

    # img = np.array(Image.open(io.BytesIO(file)))

    # if Path(file).suffix == '.pdf':
    #     images = convert_from_path(file)
    #     img = np.array(images[0])
    #     img = (rgb2gray(img)*255).astype('uint8')
    # else:
    #     img = cv2.imread(file, 0)
    file_bytes = np.asarray(bytearray(file), dtype=np.uint8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    h, w,  = img.shape

    # Remove upper and lower part of image + left and right margin.
    img = img[int(h*0.23):int(h*0.834), int(w*0.03):int(w*0.955)]

    # Binarize and invert.
    thresh, img_bin = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
    img_bin = 255-img_bin

    # Detect vertical lines of at least 1/60 of total height.
    v_kernel_len = np.array(img).shape[1]//60
    # Detect horizontal lines of at least 1/20 of total width.
    h_kernel_len = np.array(img).shape[0]//20
    # Define a vertical kernel to detect all vertical lines of image.
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    # Define a horizontal kernel to detect all horizontal lines of image.
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))

    # Use vertical kernel to detect the vertical lines.
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)

    # Use horizontal kernel to detect the horizontal lines.
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    # Theshold and skeletonize the image.
    thresh, img_vh = cv2.threshold(
        img_vh, 128, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_vh = skeletonize(img_vh)

    # Extract row indices.
    single_col = img_vh[:, img_vh.shape[1]//6]
    row_indices = np.nonzero(single_col)[0]
    # calculate 1/3 of row height for detection of descenders of characters
    descenders_height = int((row_indices[3] - row_indices[2]) * 0.33)

    # Extract column indices from middle of arbitrary row.
    single_row = img_vh[(row_indices[3]+row_indices[4])//2]
    col_indices = np.nonzero(single_row)[0]
    col_indices = np.hstack((0, col_indices, img_vh.shape[1]-1))

    result = np.empty((len(row_indices)-2, len(col_indices)-1), dtype=object)

    for i in range(len(row_indices)-2):
        for j in range(len(col_indices)-1):
            # For column with object names, extend lower boundary of cell.
            if j == 0:
                img_patch = img[row_indices[i]+1:row_indices[i+1] +
                                descenders_height, col_indices[j]+1:col_indices[j+1]]
            else:
                img_patch = img[row_indices[i]+1:row_indices[i+1],
                                col_indices[j]+1:col_indices[j+1]]
            result[i, j] = img_patch

    return result


def recognize_cells(cells):
    results = []
    for row in cells:
        results.append([])
        for pic in row:
            img = Image.fromarray(pic)
            txt = pytesseract.image_to_string(
                img, lang="Model_3_2209+Model_4_2401+Model_deu2+model3+model4")
            results[-1].append((img, txt))
    return results


def save_to_docx(recognized_cells):
    return recognized_images_to_docx(recognized_cells)


def recognized_images_to_docx(recognized_images):
    def img_to_io(img) -> BytesIO:
        with BytesIO() as output:
            img.save(output, format="JPEG")
            r = BytesIO(output.getvalue())
        return r

    document = Document()

    max_i = len(recognized_images)
    max_j = len(recognized_images[0])

    table = document.add_table(rows=max_i, cols=max_j)
    table.style = 'Table Grid'
    print("started looping ")
    for i in range(max_i):
        for j in range(max_j):
            _img, text = recognized_images[i][j]
            img = img_to_io(_img)

            cell = table.rows[i].cells[j]
            paragraph = cell.paragraphs[0]
            run = paragraph.add_run()
            run.add_picture(img)
            cell.add_paragraph(text)

    result = BytesIO()
    document.save(result)
    result.seek(0)

    print("here is the doc ", document)
    return result


def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet.
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())


# cells = pdf_to_cells('Bsp 1-5_geschwärzt-2-1.png')
# recognized_cells = recognize_cells(cells)
# print("starting saving ")

# file = save_to_docx(recognized_cells)
# write_bytesio_to_file('test.docx', file)
