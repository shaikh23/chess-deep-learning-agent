"""Interactive chess GUI to play against the trained AI model.

Usage:
    python play_gui.py --model artifacts/weights/best_model.pth

Controls:
    - Click piece, then click destination square to move
    - ESC: Quit
    - R: Reset game
    - U: Undo last move
"""

import pygame
import chess
import torch
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.model.nets import MiniResNetPolicyValue
from src.play.engine_wrapper import NeuralEngineWrapper

# Colors
WHITE = (240, 240, 240)
BLACK = (50, 50, 50)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT_COLOR = (186, 202, 68)
SELECTED_COLOR = (246, 246, 130)
TEXT_COLOR = (50, 50, 50)
BUTTON_COLOR = (100, 149, 237)
BUTTON_HOVER = (70, 130, 220)

# Board dimensions
SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8
INFO_PANEL_WIDTH = 300
WINDOW_WIDTH = BOARD_SIZE + INFO_PANEL_WIDTH
WINDOW_HEIGHT = BOARD_SIZE + 100

# Piece image filenames mapping
PIECE_IMAGE_NAMES = {
    'P': 'white_pawn', 'N': 'white_knight', 'B': 'white_bishop',
    'R': 'white_rook', 'Q': 'white_queen', 'K': 'white_king',
    'p': 'black_pawn', 'n': 'black_knight', 'b': 'black_bishop',
    'r': 'black_rook', 'q': 'black_queen', 'k': 'black_king',
}

# Standard chess piece values
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # King has no point value
}


class ChessGUI:
    """Interactive chess GUI with AI opponent."""

    def __init__(self, model_path: str, player_color: str = "white", search_depth: int = 3):
        """Initialize the chess GUI.

        Args:
            model_path: Path to trained model weights
            player_color: "white" or "black"
            search_depth: AI search depth (higher = stronger but slower)
        """
        pygame.init()

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess AI - Play Against Your Model")

        # Initialize fonts
        self.info_font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

        # Load piece images
        self.piece_images = {}
        self._load_piece_images()

        # Game state
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.player_color = chess.WHITE if player_color.lower() == "white" else chess.BLACK
        self.game_over = False
        self.status_message = "Your turn!"
        self.move_history = []

        # Load AI model
        print(f"Loading model from {model_path}...")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = MiniResNetPolicyValue(num_blocks=6, channels=64)
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

        # Initialize AI engine
        self.ai_engine = NeuralEngineWrapper(
            self.model,
            self.device,
            search_type="alphabeta",
            search_config={"max_depth": search_depth, "time_limit": 5.0}
        )

        print(f"Playing as {player_color}")
        print(f"AI search depth: {search_depth}")

        # If AI plays first (player is black), make AI move
        if self.player_color == chess.BLACK:
            self.make_ai_move()

    def get_material_count(self) -> Tuple[int, int]:
        """Calculate material count for both sides.

        Returns:
            Tuple of (white_material, black_material) point counts
        """
        white_material = 0
        black_material = 0

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = PIECE_VALUES[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value

        return white_material, black_material

    def _load_piece_images(self):
        """Load chess piece images from the pieces directory."""
        pieces_dir = Path(__file__).parent / "pieces"

        if not pieces_dir.exists():
            print(f"Warning: pieces directory not found at {pieces_dir}")
            print("Run 'python download_chess_pieces_png.py' to download piece images")
            return

        for piece_symbol, image_name in PIECE_IMAGE_NAMES.items():
            image_path = pieces_dir / f"{image_name}.png"

            if image_path.exists():
                try:
                    # Load and scale image to fit square
                    image = pygame.image.load(str(image_path))
                    # Scale to 90% of square size for nice padding
                    scaled_size = int(SQUARE_SIZE * 0.9)
                    image = pygame.transform.smoothscale(image, (scaled_size, scaled_size))
                    self.piece_images[piece_symbol] = image
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
            else:
                print(f"Warning: Image not found: {image_path}")

    def get_square_from_mouse(self, pos: Tuple[int, int]) -> Optional[int]:
        """Convert mouse position to chess square index.

        Args:
            pos: (x, y) mouse position

        Returns:
            Square index (0-63) or None if outside board
        """
        x, y = pos
        if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
            return None

        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)
        return chess.square(file, rank)

    def draw_board(self):
        """Draw the chess board."""
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                x = file * SQUARE_SIZE
                y = (7 - rank) * SQUARE_SIZE

                # Determine square color
                is_light = (rank + file) % 2 == 0
                color = LIGHT_SQUARE if is_light else DARK_SQUARE

                # Highlight selected square
                if square == self.selected_square:
                    color = SELECTED_COLOR
                # Highlight legal move destinations
                elif self.selected_square is not None:
                    for move in self.legal_moves:
                        if move.to_square == square:
                            color = HIGHLIGHT_COLOR
                            break

                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

        # Draw coordinate labels
        self.draw_coordinates()

    def draw_pieces(self):
        """Draw the chess pieces."""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                x = file * SQUARE_SIZE
                y = (7 - rank) * SQUARE_SIZE

                piece_symbol = piece.symbol()

                # Use image if available, otherwise skip (images should always be loaded)
                if piece_symbol in self.piece_images:
                    image = self.piece_images[piece_symbol]
                    # Center the image in the square
                    image_rect = image.get_rect(center=(x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2))
                    self.screen.blit(image, image_rect)

    def draw_coordinates(self):
        """Draw rank and file labels on the board edges."""
        coord_font = pygame.font.Font(None, 20)

        # Draw file labels (a-h) at the bottom
        for file in range(8):
            file_letter = chr(ord('a') + file)
            x = file * SQUARE_SIZE + SQUARE_SIZE // 2
            y = BOARD_SIZE - 8  # Bottom edge

            # Use light color on dark squares, dark color on light squares
            is_light = file % 2 == 0
            color = DARK_SQUARE if is_light else LIGHT_SQUARE

            text = coord_font.render(file_letter, True, color)
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)

        # Draw rank labels (1-8) on the left
        for rank in range(8):
            rank_number = str(rank + 1)
            x = 8  # Left edge
            y = (7 - rank) * SQUARE_SIZE + SQUARE_SIZE // 2

            # Use light color on dark squares, dark color on light squares
            is_light = rank % 2 == 0
            color = DARK_SQUARE if is_light else LIGHT_SQUARE

            text = coord_font.render(rank_number, True, color)
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)

    def draw_info_panel(self):
        """Draw the information panel on the right side."""
        # Background
        panel_x = BOARD_SIZE
        pygame.draw.rect(self.screen, WHITE, (panel_x, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT))

        y_offset = 20

        # Title
        title = self.info_font.render("Chess AI", True, TEXT_COLOR)
        self.screen.blit(title, (panel_x + 20, y_offset))
        y_offset += 50

        # Game status
        status_lines = self.status_message.split('\n')
        for line in status_lines:
            status = self.small_font.render(line, True, TEXT_COLOR)
            self.screen.blit(status, (panel_x + 20, y_offset))
            y_offset += 30

        y_offset += 20

        # Move count
        move_num = len(self.board.move_stack)
        moves_text = self.small_font.render(f"Moves: {move_num}", True, TEXT_COLOR)
        self.screen.blit(moves_text, (panel_x + 20, y_offset))
        y_offset += 30

        # Player info
        player_text = "You: " + ("White" if self.player_color == chess.WHITE else "Black")
        player_render = self.small_font.render(player_text, True, TEXT_COLOR)
        self.screen.blit(player_render, (panel_x + 20, y_offset))
        y_offset += 30

        ai_text = "AI: " + ("Black" if self.player_color == chess.WHITE else "White")
        ai_render = self.small_font.render(ai_text, True, TEXT_COLOR)
        self.screen.blit(ai_render, (panel_x + 20, y_offset))
        y_offset += 40

        # Material count
        white_material, black_material = self.get_material_count()
        material_diff = white_material - black_material

        # Determine who's ahead and by how much
        if material_diff > 0:
            advantage_text = f"White: +{material_diff}"
            advantage_color = (50, 150, 50) if self.player_color == chess.WHITE else (150, 50, 50)
        elif material_diff < 0:
            advantage_text = f"Black: +{abs(material_diff)}"
            advantage_color = (50, 150, 50) if self.player_color == chess.BLACK else (150, 50, 50)
        else:
            advantage_text = "Material: Equal"
            advantage_color = TEXT_COLOR

        material_label = self.small_font.render("Material:", True, TEXT_COLOR)
        self.screen.blit(material_label, (panel_x + 20, y_offset))
        y_offset += 25

        advantage_render = self.small_font.render(advantage_text, True, advantage_color)
        self.screen.blit(advantage_render, (panel_x + 20, y_offset))
        y_offset += 35

        # Last move
        if self.board.move_stack:
            last_move = self.board.move_stack[-1]
            last_move_text = f"Last: {last_move.uci()}"
            last_render = self.small_font.render(last_move_text, True, TEXT_COLOR)
            self.screen.blit(last_render, (panel_x + 20, y_offset))
        y_offset += 50

        # Controls
        controls = [
            "Controls:",
            "Click to move",
            "R - Reset",
            "U - Undo",
            "ESC - Quit"
        ]

        for control in controls:
            control_render = self.small_font.render(control, True, TEXT_COLOR)
            self.screen.blit(control_render, (panel_x + 20, y_offset))
            y_offset += 25

    def draw_game_over(self):
        """Draw game over overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE))
        overlay.set_alpha(200)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))

        # Game over text
        result = self.board.result()
        if result == "1-0":
            message = "White wins!"
        elif result == "0-1":
            message = "Black wins!"
        else:
            message = "Draw!"

        text = self.info_font.render(message, True, WHITE)
        text_rect = text.get_rect(center=(BOARD_SIZE // 2, BOARD_SIZE // 2 - 30))
        self.screen.blit(text, text_rect)

        # Restart prompt
        prompt = self.small_font.render("Press R to restart", True, WHITE)
        prompt_rect = prompt.get_rect(center=(BOARD_SIZE // 2, BOARD_SIZE // 2 + 30))
        self.screen.blit(prompt, prompt_rect)

    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click on board.

        Args:
            pos: (x, y) mouse position
        """
        if self.game_over or self.board.turn != self.player_color:
            return

        square = self.get_square_from_mouse(pos)
        if square is None:
            return

        # If no piece selected, select this square if it has player's piece
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.player_color:
                self.selected_square = square
                self.legal_moves = [m for m in self.board.legal_moves
                                   if m.from_square == square]
        else:
            # Try to make a move
            move = None
            for m in self.legal_moves:
                if m.to_square == square:
                    move = m
                    break

            if move:
                # Handle promotion (default to queen)
                if move.promotion is None and self.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                    if chess.square_rank(square) in [0, 7]:
                        move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)

                self.make_move(move)
                self.selected_square = None
                self.legal_moves = []

                # AI's turn
                if not self.game_over:
                    self.make_ai_move()
            else:
                # Clicked on another piece or empty square, reselect
                piece = self.board.piece_at(square)
                if piece and piece.color == self.player_color:
                    self.selected_square = square
                    self.legal_moves = [m for m in self.board.legal_moves
                                       if m.from_square == square]
                else:
                    self.selected_square = None
                    self.legal_moves = []

    def make_move(self, move: chess.Move):
        """Make a move on the board.

        Args:
            move: The move to make
        """
        self.board.push(move)
        self.move_history.append(move)

        if self.board.is_game_over():
            self.game_over = True
            self.status_message = "Game Over!"

    def make_ai_move(self):
        """Let the AI make a move."""
        self.status_message = "AI is thinking..."
        pygame.display.flip()

        try:
            move = self.ai_engine.get_move(self.board)
            self.make_move(move)
            self.status_message = f"AI played: {move.uci()}"

            if not self.game_over:
                self.status_message += "\nYour turn!"
        except Exception as e:
            print(f"AI error: {e}")
            self.status_message = f"AI error: {str(e)}"

    def reset_game(self):
        """Reset the game to initial position."""
        self.board.reset()
        self.selected_square = None
        self.legal_moves = []
        self.game_over = False
        self.status_message = "Game reset!"
        self.move_history = []

        # If AI plays first, make AI move
        if self.player_color == chess.BLACK:
            self.make_ai_move()
        else:
            self.status_message = "Your turn!"

    def undo_move(self):
        """Undo the last move (player's move and AI's move)."""
        if len(self.board.move_stack) >= 2 and not self.game_over:
            # Undo AI's move
            self.board.pop()
            # Undo player's move
            self.board.pop()
            self.status_message = "Move undone. Your turn!"
            self.selected_square = None
            self.legal_moves = []

    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_u:
                        self.undo_move()

            # Draw everything
            self.screen.fill(WHITE)
            self.draw_board()
            self.draw_pieces()
            self.draw_info_panel()

            if self.game_over:
                self.draw_game_over()

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Play chess against your trained AI")
    parser.add_argument(
        "--model",
        type=str,
        default="artifacts/weights/best_model.pth",
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--color",
        type=str,
        default="white",
        choices=["white", "black"],
        help="Your color (white or black)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="AI search depth (higher = stronger but slower)"
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        print("\nAvailable models:")
        weights_dir = Path("artifacts/weights")
        if weights_dir.exists():
            for model_file in weights_dir.glob("*.pth"):
                print(f"  - {model_file}")
        return

    # Start GUI
    gui = ChessGUI(args.model, args.color, args.depth)
    gui.run()


if __name__ == "__main__":
    main()
