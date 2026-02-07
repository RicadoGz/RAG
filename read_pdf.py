from docx import Document

doc = Document("document/docx/MSFT_FY25q4_10K.docx")

# 段落
paras = [p.text for p in doc.paragraphs if p.text.strip()]
# this will filter out empty paragraphs and those that contain only whitespace characters.
# 表格
tables = []
for t in doc.tables:
    rows = []
    for r in t.rows:
        rows.append([c.text.strip() for c in r.cells])
    tables.append(rows)

print("paragraphs:", len(paras))
print("tables:", len(tables))

with open("output.txt", "w", encoding="utf-8") as f:
    for p in paras:
        f.write(p + "\n")
    f.write("\n\n=== TABLES ===\n")
    for ti, table in enumerate(tables, 1):
        f.write(f"\n[TABLE {ti}]\n")
        for row in table:
            f.write("\t".join(row) + "\n")


def extract_text_from_docx(file_path):
    doc = Document(file_path) #read from the paragra

