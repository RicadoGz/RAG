from docx import Document

doc = Document("document/docx/MSFT_FY25q4_10K.docx")


def extract_text_from_docx(file_path):
    doc = Document(file_path) #read from the paragra
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    # the strip means get all the space out 
    # for each line of paragrap add into paras in to a list 

    tables=[]
    for t in doc.tables:
        #this will get all the table 
        rows=[] #一行
        for r in t.rows:
            rows.append([c.text.strip() for c in r.cells])
            # this was each cell content add into each sell in rows
        tables.append(rows)
        # this will add all the rows into tables


def copyExistIntoFileTest(para: list,tables:list,file_path:str)->bool:
    try:
        with open (file_path,"w",encoding="utf-8") as f:
            for p in para:
                f.write(p +"\n") # this will only write the paragraph into the file without table
            f.write("\n\n=== TABLES ===\n")# this is write the table title into the file
            for ti,table in enumerate(tables,1):
                f.write(f"\n[TABLE {ti}]\n") # this will write the table number into the file
                for row in table:
                    f.write("\t".join(row) + "\n") # this will write the table content into the file with tab space between each cell
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False
    


def main():
    file_path = "document/docx/MSFT_FY25q4_10K.docx"
    output_path = "output.txt"
    paras, tables = extract_text_from_docx(file_path)
    success = copyExistIntoFileTest(paras, tables, output_path)
    if success:
        print(f"Content successfully written to {output_path}")
    else:
        print("Failed to write content to file.")   
if __name__ == "__main__":
            main()