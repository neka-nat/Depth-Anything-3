import zenoh


session = zenoh.open(zenoh.Config())
pub = session.declare_publisher("prompt")

while True:
    prompt = input("Enter a prompt: ")
    pub.put(prompt.encode())

session.close()
