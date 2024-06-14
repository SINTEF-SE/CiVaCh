from fpdf import FPDF
from PIL import Image
import colorFilters as cf

title = 'Kamera Rapport'
A4_width = 210
margin = 10

class PDF(FPDF):
    def header(self):
        # Logo
        self.image('civach_logo.jpg', 10, 8, 33)    # The logo in upper left corner
        # Arial bold 15
        self.set_font('helvetica', 'B', 15)
        # Calculate width of title and position
        w = self.get_string_width(title) + 6
        self.set_x((210 - w) / 2)
        # Colors of frame, background and text
        self.set_draw_color(0, 0, 0)
       # self.set_fill_color(230, 230, 0)
       # self.set_text_color(220, 50, 50)
        # Thickness of frame (0.5 mm)
        self.set_line_width(0.5)
        # Title
        self.cell(w, 9, title, 0, 1, 'C', 0)
        # Line break
        self.ln(10)
        self.line(10, 20, 200, 20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('helvetica', 'I', 8)
        # Text color in gray
        self.set_text_color(128)
        # Page number
        self.cell(0, 10, 'Side ' + str(self.page_no()), 0, 0, 'C')

    #--- printImage ----
    # Prints the two images at the top of the page
    # orgIm is the original image and boxIm is the same image with boxes around detected objects
    # This function also adds a page and must be called first
    def printImage2(self, orgIm, boxIm):
        self.add_page()
        im = Image.open(orgIm)
        width, height = im.size
        width_mm = A4_width/2-1.5*margin
        height_mm = width_mm/width*height
        self.image(orgIm, x=None, y=25, w=A4_width/2-1.5*margin, h=0)
        self.image(boxIm, x=A4_width/2+margin/2, y=25, w=A4_width/2-1.5*margin, h=0)
        return height_mm
        
    #--- printImage ----
    # Prints the two images at the top of the page
    # orgIm is the original image and boxIm is the same image with boxes around detected objects
    # This function also adds a page and must be called first
    def printImage3(self, orgIm, boxIm, segIm):
        self.add_page()
        im = Image.open(orgIm)
        width, height = im.size
        width_mm = A4_width/2-1.5*margin
        height_mm = width_mm/width*height
        self.image(orgIm, x=None, y=25, w=A4_width/2-1.5*margin, h=0)
        self.image(boxIm, x=A4_width/2+margin/2, y=25, w=A4_width/2-1.5*margin, h=0)
        self.image(segIm, x=A4_width/2 - width_mm/2, y=30 + height_mm, w=A4_width/2-1.5*margin, h=0)
        return 2*height_mm

    #--- printHeadLine ----
    # Prints the headline with categories and a thick line beneath
    # H1 to h5 are the 5 category texts
    def printHeadLine(self, h1, h2, h3, h4, h5, h6):
        self.set_font('helvetica', 'B', 10)
        w = (A4_width-2*margin)/6
        self.cell(w, 0, h1, 0, 0, 'L')
        self.cell(w, 0, h2, 0, 0, 'C')
        self.cell(w, 0, h3, 0, 0, 'C')
        self.cell(w, 0, h4, 0, 0, 'C')
        self.cell(w, 0, h5, 0, 0, 'C')
        self.cell(w, 0, h6, 0, 1, 'C')
        self.line(10, self.get_y()+3, 200, self.get_y()+3)
        self.ln(6)

     #--- printObjectLine ----
    # Prints a line for one object with all the categories texts
    # o1 to o5 are texts in each category
    # A rectangle is also printed at category 1 in the color marked by the color tuple input parameter
    def printObjectLine(self, o1, o2, o3, o4, o5, o6, color):
        self.ln(1)
        self.set_font('helvetica', '', 8)
        b,g,r = color
        self.set_fill_color(230, 230, 230)
        w = (A4_width-2*margin)/6
        self.set_draw_color(r, g, b)
        self.set_line_width(0.5)
        self.cell(w, 5, o1, 0, 0, 'L', 1)
        self.set_fill_color(r, g, b)
        self.rect(20, self.get_y()+1, 7, 3, 'DF')
        self.set_fill_color(230, 230, 230)
        self.cell(w, 5, o2, 0, 0, 'C', 1)
        self.cell(w, 5, o3, 0, 0, 'C', 1)
        self.cell(w, 5, o4, 0, 0, 'C', 1)
        self.cell(w, 5, o5, 0, 0, 'C', 1)
        self.cell(w, 5, o6, 0, 1, 'C', 1)
        
     

