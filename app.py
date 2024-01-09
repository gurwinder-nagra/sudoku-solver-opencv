import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import os, random
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


model = load_model('sudoku_net.h5')

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (3,3),6) 
    #blur = cv2.bilateralFilter(gray,9,75,75)
    threshold_img = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    return threshold_img

def main_outline(contour):
    biggest = np.array([])
    max_area = 0
    for i in contour:
        area = cv2.contourArea(i)
        if area >50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i , 0.02* peri, True)
            if area > max_area and len(approx) ==4:
                biggest = approx
                max_area = area
    return biggest ,max_area

def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4,1,2),dtype = np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis =1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new


def splitcells(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes


def CropCell(cells):
    Cells_croped = []
    for image in cells:
        
        img = np.array(image)
        img = img[4:46, 6:46]
        img = Image.fromarray(img)
        Cells_croped.append(img)
        
    return Cells_croped


def read_cells(cell,model):

    result = []
    for image in cell:
        # preprocess the image as it was in the model 
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (32, 32))
        img = img / 255
        img = img.reshape(1, 32, 32, 1)
        # getting predictions and setting the values if probabilities are above 65% 
        
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=1)
        probabilityValue = np.amax(predictions)
        
        if probabilityValue > 0.65:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result



#This function finds the next box to solve 

def find_empty(board):
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 0:
               return i,j
    return None

#Function to fill in the possible values by evaluating rows collumns and smaller cells

def valid(board, num, pos):
    # check row
    for i in range(len(board)):
        if board[pos[0]][i]==num and pos[1]!=i:
                 return False
    
    
    # check column
    for i in range(len(board)):
        if board[i][pos[1]]==num and pos[0]!=i:
            return False
        
   
        
    # check 3X3 cube
    box_x=pos[0]//3
    box_y=pos[1]//3
    
    for i in range(box_x*3,box_x+3):
        for j in range(box_y*3,box_y+3):
            if board[i][j]==num and [i,j]!=pos:
                return False
   
    return True



#function to loop over untill a valid answer is found. 

def solve(board):
    find=find_empty(board)
    if not find:
        return True
    else:
        row,col=find
        
    
    for i in range(1,10):
        if valid(board,i,[row,col]):
            board[row][col]=i
            
            if solve(board):
                return True
            
            board[row][col]=0
    return False


def display(quiz):
    for row in range(9):
        if row % 3 == 0 and row != 0:
            print("....................")

        for col in range(9):
            if col % 3 == 0 and col != 0:
                print("|", end=" ")

            if col == 8:
                print(quiz[row][col])
            else:
                print(str(quiz[row][col]) + " ", end="")

    
def main():
    st.title('Sudoku Solver')

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        col1.subheader("Sudoku Puzzle:")
        puzzle = Image.open(uploaded_file)
        col1.image(puzzle, use_column_width=True)
        puzzle = np.asarray(puzzle)
        # Resizing puzzle to be solved
        puzzle = cv2.resize(puzzle, (450,450))
        # Preprocessing Puzzle 
        su_puzzle = preprocess(puzzle)
        
        # Finding the outline of the sudoku puzzle in the image
        su_contour_1= su_puzzle.copy()
        su_contour_2= puzzle.copy()
        su_contour, hierarchy = cv2.findContours(su_puzzle,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(su_contour_1, su_contour,-1,(0,255,0),3)
        
        black_img = np.zeros((450,450,3), np.uint8)
        su_biggest, su_maxArea = main_outline(su_contour)
        if su_biggest.size != 0:
            su_biggest = reframe(su_biggest)
            cv2.drawContours(su_contour_2,su_biggest,-1, (0,255,0),10)
            su_pts1 = np.float32(su_biggest)
            su_pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
            su_matrix = cv2.getPerspectiveTransform(su_pts1,su_pts2)  
            su_imagewrap = cv2.warpPerspective(puzzle,su_matrix,(450,450))
            su_imagewrap =cv2.cvtColor(su_imagewrap, cv2.COLOR_BGR2GRAY)
            _, su_imagewrap = cv2.threshold(su_imagewrap, 127, 255, cv2.THRESH_BINARY)

        sudoku_cell = splitcells(su_imagewrap)
        sudoku_cell_croped= CropCell(sudoku_cell)

        grid = read_cells(sudoku_cell_croped, model)
        grid = np.asarray(grid)
        grid = np.reshape(grid,(9,9))
        solve(grid)
    
        # Display the solution or appropriate message
         
        if solve(grid):
           
            col2.subheader("Sudoku Solved:")
            col2.table(grid)
        
        else:
            st.write("No Solution Found.")


if __name__ == '__main__':
    main()