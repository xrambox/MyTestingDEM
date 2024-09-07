import turtle
import random

# Set up the screen
screen = turtle.Screen()
screen.bgcolor("black")

# Create a turtle object
t = turtle.Turtle()
t.speed(0)  # Set the speed of the turtle to the fastest
t.width(2)  # Set the pen width

# Function to draw a heart
def draw_heart():
    t.color(random.choice(['red', 'pink', 'purple', 'yellow', 'blue']))  # Set random color of the heart
    t.begin_fill()  # Start filling the heart shape
    t.left(50)
    t.forward(133)
    t.circle(50, 200)  # Draw the left curve of the heart
    t.right(140)
    t.circle(50, 200)  # Draw the right curve of the heart
    t.forward(133)
    t.end_fill()  # End filling
    t.setheading(0)  # Reset direction

# Function to write text in the center heart
def write_message():
    t.penup()
    t.goto(0, 20)
    t.color("white")
    t.write("LISSA I MISS YOU", align="center", font=("Arial", 24, "bold"))
    t.goto(0, 0)

# Function to add shading for 3D effect (Optional)
def add_shading():
    t.penup()
    t.goto(5, -105)  # Move slightly to create a shadow effect
    t.pendown()
    t.color("darkred")
    t.setheading(50)
    t.forward(133)
    t.circle(50, 200)
    t.right(140)
    t.circle(50, 200)
    t.forward(133)

# Draw 100 hearts in random positions
for _ in range(100):
    t.penup()
    x = random.randint(-300, 300)
    y = random.randint(-300, 300)
    t.goto(x, y)
    t.pendown()
    
    # Draw the heart
    draw_heart()

# Draw one big heart at the center with the message
t.penup()
t.goto(0, -100)
t.pendown()
draw_heart()
add_shading()
write_message()

# Hide the turtle
t.hideturtle()

# Keep the window open until it's closed manually
turtle.done()
