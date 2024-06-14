import xlsxwriter as xl
import os


def createExcel(filename, dirname, method):
    lastDir = dirname.split("/")[-1] #only text after /
    fileList = os.listdir(dirname)
    noFiles = len(fileList)
    wb = xl.Workbook(filename)
    ws = wb.add_worksheet(lastDir)
    fbold_wrap = wb.add_format({'bold': True, 'text_wrap': True})
    fbigText = wb.add_format({'font_size': 16})
    #file info
    ws.write('A1', 'Image directory: ' + dirname + ',  Number of images: '+ str(noFiles) + ', Segment method: ' + method, fbigText )
    #heading
    ws.write('A2', 'Segmentering', fbold_wrap)
    ws.write('E2', 'Farge', fbold_wrap)
    ws.write('F2', 'Klassifisering', fbold_wrap)
    ws.write('I2', 'Forurensning', fbold_wrap)
    ws.write('L2', 'Tid', fbold_wrap)
   
    #line 2
    ws.write('A3', 'Bilde nr.', fbold_wrap)
    ws.write('B3', 'TP - Rigktig seg. objekt', fbold_wrap)
    ws.write('C3', 'FN - Manglende seg. objekt',fbold_wrap)
    ws.write('D3', 'FP - Ghost seg. objekt',fbold_wrap)
    ws.write('E3', 'Antall feil farge', fbold_wrap)
    ws.write('F3', 'TP - Rigktig klass. objekt', fbold_wrap)
    ws.write('G3', 'FN - Manglende eller feil klass. objekt', fbold_wrap)
    ws.write('H3', 'FP - Ghost klass. objekt', fbold_wrap)
    ws.write('I3', 'Manuell antatt forurensning', fbold_wrap)
    ws.write('J3', 'Forurensningsgrad', fbold_wrap)
    ws.write('K3', 'Gj. snitt avvik forurensningsgrad', fbold_wrap)
    ws.write('L3', 'Tid segmentering', fbold_wrap)
    ws.write('M3', 'Tid klassifisering', fbold_wrap)
    ws.write('N3', 'Total tid', fbold_wrap)
    

    #formulas after end of file listing
    row = noFiles + 4
    rowIdx = 2 + noFiles + 1
    #TP seg
    col = 1
    formula = '=sum(B4:B%d)' %rowIdx
    ws.write_formula(row, col, formula)
    #FN seg
    col += 1
    formula = '=sum(C4:C%d)' %rowIdx
    ws.write_formula(row, col, formula)
    #FP seg
    col += 1
    formula = '=sum(D4:D%d)' %rowIdx
    ws.write_formula(row, col, formula)
    #antall farge
    col += 1
    formula = '=sum(E4:E%d)' %rowIdx
    ws.write_formula(row, col, formula)
    #TP class
    col += 1
    formula = '=sum(F4:F%d)' %rowIdx
    ws.write_formula(row, col, formula)
    #FN class
    col += 1
    formula = '=sum(G4:G%d)' %rowIdx
    ws.write_formula(row, col, formula)
    #FP class
    col += 1
    formula = '=sum(H4:H%d)' %rowIdx
    ws.write_formula(row, col, formula)
    
    col += 3
    formula = '=average(K4:K%d)' %rowIdx
    ws.write_formula(row, col, formula)
    col += 1
    formula = '=average(L4:L%d)' %rowIdx
    ws.write_formula(row, col, formula)
    col += 1
    formula = '=average(M4:M%d)' %rowIdx
    ws.write_formula(row, col, formula)
    col += 1
    formula = '=average(N4:N%d)' %rowIdx
    ws.write_formula(row, col, formula)

    col = 10
    row += 1
    formula = '=STDEV(K4:K%d)' %rowIdx
    ws.write_formula(row, col, formula)
    col += 1
    formula = '=stdev(L4:L%d)' %rowIdx
    ws.write_formula(row, col, formula)
    col += 1
    formula = '=stdev(M4:M%d)' %rowIdx
    ws.write_formula(row, col, formula)
    col += 1
    formula = '=stdev(N4:N%d)' %rowIdx
    ws.write_formula(row, col, formula)

    rowIdx += 2
    row +=2
    col = 0
    ws.write(row, col, 'Klassifisering', fbold_wrap)
    row += 1
    ws.write(row, col, 'Presisjon', fbold_wrap)
    col += 1
    formula = '=F%d/(F%d + H%d)' %(rowIdx, rowIdx, rowIdx)
    ws.write_formula(row, col, formula)

    row +=1
    col = 0
    ws.write(row, col, 'Deteksjonsgrad', fbold_wrap)
    col += 1
    formula = '=F%d/(F%d + G%d)' %(rowIdx, rowIdx, rowIdx)
    ws.write_formula(row, col, formula)

    row += 1
    col = 0
    ws.write(row, col, 'Nøyaktighet', fbold_wrap)
    col += 1
    formula = '=F%d/(F%d + + G%d + H%d)' %(rowIdx, rowIdx, rowIdx, rowIdx)
    ws.write_formula(row, col, formula)

    row += 1
    col = 0
    ws.write(row, col, 'F-score', fbold_wrap)
    col += 1
    formula = '=2*F%d/(2*F%d + G%d + H%d)' %(rowIdx, rowIdx, rowIdx, rowIdx)
    ws.write_formula(row, col, formula)
    
    row +=2
    col = 0
    ws.write(row, col, 'Segmentering', fbold_wrap)
    row += 1
    ws.write(row, col, 'Presisjon', fbold_wrap)
    col += 1
    formula = '=B%d/(B%d + D%d)' %(rowIdx, rowIdx, rowIdx)
    ws.write_formula(row, col, formula)

    row +=1
    col = 0
    ws.write(row, col, 'Deteksjonsgrad', fbold_wrap)
    col += 1
    formula = '=B%d/(B%d + C%d)' %(rowIdx, rowIdx, rowIdx)
    ws.write_formula(row, col, formula)

    row += 1
    col = 0
    ws.write(row, col, 'Nøyaktighet', fbold_wrap)
    col += 1
    formula = '=B%d/(B%d + + C%d + D%d)' %(rowIdx, rowIdx, rowIdx, rowIdx)
    ws.write_formula(row, col, formula)

    row += 1
    col = 0
    ws.write(row, col, 'F-score', fbold_wrap)
    col += 1
    formula = '=2*B%d/(2*B%d + C%d + D%d)' %(rowIdx, rowIdx, rowIdx, rowIdx)
    ws.write_formula(row, col, formula)

    return wb, ws

