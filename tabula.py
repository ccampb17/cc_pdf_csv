# Import Module
import tabula
#from tabula.io import read_pdf
# Read PDF File
# this contain a list


df = tabula.read_pdf(r"C:\Users\c-cam\GTA data team Dropbox\Bastiat\0 projects\044 PDF to CSV\tables_only_4_11.pdf", pages = 1)[0]