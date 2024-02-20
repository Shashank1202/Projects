import pygame
from flask import Flask, render_template
import random
import threading

app = Flask(__name__)

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 400
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 60
BALL_RADIUS = 10
WHITE = (255, 255, 255)
RED = (255, 0, 0)
FONT_SIZE = 30

# Create the paddles
player_paddle = pygame.Rect(50, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
computer_paddle = pygame.Rect(WIDTH - 50 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)

# Create the ball
ball_x = WIDTH // 2
ball_y = HEIGHT // 2
ball_speed_x = random.choice([-5, 5])
ball_speed_y = random.choice([-5, 5])

# Player and opponent names
player_name = "Player"
opponent_name = "AI"

# Player and opponent scores
player_score = 0
opponent_score = 0

# Fonts
font = pygame.font.Font(None, FONT_SIZE)

# Function to run the game loop in a separate thread
def run_game():
    global ball_x, ball_y, ball_speed_x, ball_speed_y, player_score, opponent_score

    while True:
        # Move the ball
        ball_x += ball_speed_x
        ball_y += ball_speed_y

        # Bounce off the walls
        if ball_y - BALL_RADIUS < 0 or ball_y + BALL_RADIUS > HEIGHT:
            ball_speed_y = -ball_speed_y

        # Bounce off the paddles
        if player_paddle.colliderect(pygame.Rect(ball_x - BALL_RADIUS, ball_y - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)):
            ball_speed_x = -ball_speed_x

        if computer_paddle.colliderect(pygame.Rect(ball_x - BALL_RADIUS, ball_y - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)):
            ball_speed_x = -ball_speed_x

        # Move the computer paddle to follow the ball
        if computer_paddle.centery < ball_y:
            computer_paddle.y += 3
        elif computer_paddle.centery > ball_y:
            computer_paddle.y -= 3

        # Scoring
        if ball_x - BALL_RADIUS < 0:
            opponent_score += 1
            ball_x = WIDTH // 2
            ball_y = HEIGHT // 2
            ball_speed_x = random.choice([-5, 5])
            ball_speed_y = random.choice([-5, 5])

        elif ball_x + BALL_RADIUS > WIDTH:
            player_score += 1
            ball_x = WIDTH // 2
            ball_y = HEIGHT // 2
            ball_speed_x = random.choice([-5, 5])
            ball_speed_y = random.choice([-5, 5])

        pygame.time.Clock().tick(30)


@app.route('/')
def index():
    return render_template('index.html', player_name=player_name, opponent_name=opponent_name)


@app.route('/get_state')
def get_state():
    return {
        'player_score': player_score,
        'opponent_score': opponent_score,
        'ball_x': ball_x,
        'ball_y': ball_y,
        'player_paddle_y': player_paddle.y,
        'computer_paddle_y': computer_paddle.y
    }


if __name__ == '__main__':
    # Start the game loop in a separate thread
    game_thread = threading.Thread(target=run_game)
    game_thread.start()

    # Run the Flask app
    app.run(debug=True)
