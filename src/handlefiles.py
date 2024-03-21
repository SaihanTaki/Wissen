from PyPDF2 import PdfReader
import docx2txt


class File2Text:
    def __init__(
        self,
        fileobject,
    ):
        self.fileobject = fileobject
        self.filename = fileobject.name
        self.file_extension = fileobject.name.split(".")[-1]

    def __call__(self):
        if self.file_extension == "pdf":
            text = self.handlePDF()
        elif self.file_extension == "txt":
            text = self.handleTEXT()
        elif self.file_extension in ["doc", "docx", "docm"]:
            text = self.handleDOCX()
        else:
            text = "hello there is a problem"
        return text

    def handlePDF(self):
        pdf_reader = PdfReader(self.fileobject)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        return text

    def handleTEXT(self):
        text = ""
        for line in self.fileobject.readlines():
            text += str(line, "utf-8")
        return text

    def handleDOCX(self):
        text = docx2txt.process(self.fileobject)
        text = " ".join(text.split())
        return text
