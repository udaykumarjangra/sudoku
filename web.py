import streamlit as st
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image as im
import numpy as np
import imutils
import cv2 as cv
from sudoku import Sudoku

def main():
    st.title("Sudoko Solver")
    img = st.file_uploader("Please upload an image file", type = ["jpg","png"])
    if img is None:
        st.text("No image uploaded")
    else:
        img = im.open(img)
        img = np.asarray(img)
        img = imutils.resize(img, width = 600)
        prediction = import_and_predict(img)
        if prediction is not None:
            st.image(prediction)
        else:
            st.text("No solution found")

def boundaries(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7,7), 3)
    threshold = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11,2)
    threshold = cv.bitwise_not(threshold)
    contour = cv.findContours(threshold.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    contour = sorted(contour, key = cv.contourArea, reverse=True)
    count = None
    
    for c in contour:
        length = cv.arcLength(c, True)
        
        approx = cv.approxPolyDP(c, 0.02 * length, True)
        
        if len(approx) == 4:
            count = approx
            break
        
    if count is None:
        print("No puzzle found")

    img = image.copy()
    cv.drawContours(img, [count], -1, (0,255,0),2)
    puzzle = four_point_transform(image, count.reshape(4,2))
    warped = four_point_transform(gray, count.reshape(4,2)) 
    return (puzzle, warped)

def extract(cell):
    threshold = cv.threshold(cell, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    threshold = clear_border(threshold)
    contour = cv.findContours(threshold.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    img = cell.copy()
    cv.drawContours(img, contour, -1, (0,255,0),2)
    if len(contour) == 0:
        return None
    
    c = max(contour, key = cv.contourArea)
    mask = np.zeros(threshold.shape).astype("int8")
    cv.drawContours(mask, [c], -1, 255, -1)
    (height, width) = threshold.shape
    filled = cv.countNonZero(mask)/float(height*width)
    
    if filled < 0.03:
        return None
    
    digit = cv.bitwise_and(threshold, threshold, mask=mask)
    return digit

def import_and_predict(image_data):
    digit_classifier = load_model("digit_classifier")
    (color, grey) = boundaries(image_data)
    board = np.zeros((9,9), dtype = "int")
    X_size = grey.shape[1] // 9
    y_size = grey.shape[0] // 9

    cells = []

    for y in range(0,9):
        row = []
        for x in range(0,9):
            startX = x * X_size
            startY = y * y_size
            endX = (x+1) * X_size
            endY = (y+1) * y_size
            row.append((startX, startY, endX, endY))
            cell = grey[startY:endY, startX:endX]
            digit = extract(cell)
            
            if digit is not None:
                area = cv.resize(digit, (28,28))
                area = area.astype("float") / 255.0
                area = img_to_array(area)
                area = np.expand_dims(area, axis = 0)
                prediction = digit_classifier.predict(area).argmax(axis=1)[0]
                print(prediction)
                board[y,x] = prediction
        cells.append(row)
    solution = Sudoku(3,3, board = board.tolist())
    solution = solution.solve()
    print(solution.board)
    if (solution.board[0][0]==None):
        return None
    print(board)
    for (cell, sol) in zip(cells, solution.board):
        for(box, digit) in zip(cell, sol):
            X1, Y1, X2, Y2 = box
            X = int((X2-X1) * 0.33)
            Y = int((Y2-Y1) * -0.2)
            X += X1
            Y += Y2
            print(digit)
            cv.putText(color, str(digit), (X,Y), cv.FONT_HERSHEY_SIMPLEX,0.9, (0,142,255),2)
    return color

if __name__ == '__main__':
    main()
