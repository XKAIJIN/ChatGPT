from chatgpt import ChatGPT

# 初始化 ChatGPT 模型
model = ChatGPT()

# 与 ChatGPT 进行对话
response = model.generate_response("你好，你叫什么名字？")
print("ChatGPT:", response)
