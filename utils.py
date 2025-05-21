def get_words_from_text(text):
  text = text.replace(":","")
  text = text.replace("?","")
  text = text.replace("!","")
  text = text.replace(",","")
  text = text.replace(";","")
  text = text.replace("-","")
  return text.split()