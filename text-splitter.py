""" Text splitter """

from langchain.text_splitter import MarkdownTextSplitter, PythonCodeTextSplitter, RecursiveCharacterTextSplitter

python_text = """
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

p1 = Person("John", 36)

for i in range(10):
    print (i)
"""


python_splitter = PythonCodeTextSplitter(chunk_size=100, chunk_overlap=0)


documents = python_splitter.create_documents([python_text])

for doc in documents:
    print(doc.page_content)


markdown_text = """
# Fun in California

## Driving

Try driving on the 1 down to San Diego

### Food

Make sure to eat a burrito while you're there

## Hiking

Go to Yosemite
"""

print("\n-----------------------")
print("Markdown splitter")
print("-----------------------\n")

markdown_splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=0)

documents = markdown_splitter.create_documents([markdown_text])

for doc in documents:
    print(doc.page_content)

print("\n-----------------------")
print("Recursive text splitter")
print("-----------------------\n")


text = """
One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.

Teachers and coaches implicitly told us the returns were linear. "You get out," I heard a thousand times, "what you put in." They meant well, but this is rarely true. If your product is only half as good as your competitor's, you don't get half as many customers. You get no customers, and you go out of business.

It's obviously true that the returns for performance are superlinear in business. Some think this is a flaw of capitalism, and that if we changed the rules it would stop being true. But superlinear returns for performance are a feature of the world, not an artifact of rules we've invented. We see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity. In all of these, the rich get richer. [1]
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=65, chunk_overlap=0)

documents = text_splitter.create_documents([text])

for doc in documents:
    print(doc.page_content)
    print("\n")
