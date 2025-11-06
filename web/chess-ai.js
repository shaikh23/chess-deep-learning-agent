// Chess AI Web App - Main JavaScript
// Integrates chess.js, chessboard.js, and ONNX Runtime

class ChessAI {
    constructor() {
        this.game = new Chess();
        this.board = null;
        this.session = null;
        this.playerColor = 'white';
        this.isThinking = false;
        this.moveHistory = [];
        this.pendingPromotion = null;

        this.init();
    }

    async init() {
        this.setupBoard();
        this.setupControls();
        await this.loadModel();
        this.updateStatus('Ready to play! Make your move.');
        this.updateTurnIndicator();

        // If player is black, AI moves first
        if (this.playerColor === 'black') {
            setTimeout(() => this.makeAIMove(), 500);
        }
    }

    setupBoard() {
        const config = {
            draggable: true,
            position: 'start',
            onDragStart: this.onDragStart.bind(this),
            onDrop: this.onDrop.bind(this),
            onSnapEnd: this.onSnapEnd.bind(this)
        };

        this.board = Chessboard('board', config);

        // Responsive board
        window.addEventListener('resize', () => {
            this.board.resize();
        });
    }

    setupControls() {
        // Reset button
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetGame();
        });

        // Undo button
        document.getElementById('undo-btn').addEventListener('click', () => {
            this.undoMove();
        });

        // Flip button
        document.getElementById('flip-btn').addEventListener('click', () => {
            this.board.flip();
        });

        // Color selector
        document.getElementById('color-select').addEventListener('change', (e) => {
            this.playerColor = e.target.value;
            this.resetGame();
        });

        // Promotion modal
        document.querySelectorAll('.promotion-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const piece = e.target.dataset.piece;
                this.handlePromotion(piece);
            });
        });
    }

    async loadModel() {
        this.updateStatus('Loading AI model...');

        try {
            // Create ONNX Runtime session
            this.session = await ort.InferenceSession.create('model.onnx', {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            console.log('Model loaded successfully!');
            console.log('Input names:', this.session.inputNames);
            console.log('Output names:', this.session.outputNames);

        } catch (error) {
            console.error('Error loading model:', error);
            this.updateStatus('Error loading model. Please check console.');
        }
    }

    onDragStart(source, piece, position, orientation) {
        // Don't allow moves if game is over
        if (this.game.game_over()) return false;

        // Don't allow moves if AI is thinking
        if (this.isThinking) return false;

        // Don't allow moves if it's not player's turn
        const turn = this.game.turn();
        if ((turn === 'w' && this.playerColor === 'black') ||
            (turn === 'b' && this.playerColor === 'white')) {
            return false;
        }

        // Only pick up pieces for the player's color
        if ((this.playerColor === 'white' && piece.search(/^b/) !== -1) ||
            (this.playerColor === 'black' && piece.search(/^w/) !== -1)) {
            return false;
        }
    }

    onDrop(source, target) {
        // Check if move is a pawn promotion
        const moves = this.game.moves({ verbose: true });
        const move = moves.find(m => m.from === source && m.to === target);

        if (move && move.flags.includes('p')) {
            // Store pending promotion
            this.pendingPromotion = { from: source, to: target };
            this.showPromotionModal();
            return 'snapback';
        }

        // Try to make the move
        const moveResult = this.game.move({
            from: source,
            to: target,
            promotion: 'q' // Default to queen if somehow promotion slips through
        });

        // Illegal move
        if (moveResult === null) return 'snapback';

        // Update UI
        this.addMoveToHistory(moveResult);
        this.updateStatus('AI is thinking...');

        // Check for game over
        if (this.checkGameOver()) return;

        // AI's turn
        setTimeout(() => this.makeAIMove(), 500);
    }

    onSnapEnd() {
        this.board.position(this.game.fen());
    }

    showPromotionModal() {
        document.getElementById('promotion-modal').style.display = 'block';
    }

    hidePromotionModal() {
        document.getElementById('promotion-modal').style.display = 'none';
    }

    handlePromotion(piece) {
        if (!this.pendingPromotion) return;

        const move = this.game.move({
            from: this.pendingPromotion.from,
            to: this.pendingPromotion.to,
            promotion: piece
        });

        this.hidePromotionModal();
        this.pendingPromotion = null;

        if (move === null) {
            this.board.position(this.game.fen());
            return;
        }

        this.board.position(this.game.fen());
        this.addMoveToHistory(move);
        this.updateStatus('AI is thinking...');

        if (this.checkGameOver()) return;

        setTimeout(() => this.makeAIMove(), 500);
    }

    async makeAIMove() {
        if (this.game.game_over()) return;

        this.isThinking = true;
        const startTime = Date.now();

        try {
            // Get AI move
            const move = await this.getAIMove();

            if (move) {
                this.game.move(move);
                this.board.position(this.game.fen());
                this.addMoveToHistory(move);

                const thinkTime = ((Date.now() - startTime) / 1000).toFixed(2);
                document.getElementById('think-time').textContent = `${thinkTime}s`;

                this.updateStatus('Your turn!');
                this.checkGameOver();
            } else {
                this.updateStatus('AI error: No legal move found');
            }
        } catch (error) {
            console.error('AI move error:', error);
            this.updateStatus('AI error occurred');
        }

        this.isThinking = false;
        this.updateTurnIndicator();
    }

    async getAIMove() {
        // Encode board position
        const boardTensor = this.encodeBoard();
        const legalMask = this.getLegalMovesMask();

        // Run inference
        const feeds = {
            board: boardTensor,
            legal_mask: legalMask
        };

        const results = await this.session.run(feeds);
        const policyLogits = results.policy_logits.data;
        const value = results.value.data[0];

        // Update evaluation display
        document.getElementById('evaluation').textContent = value.toFixed(2);

        // Get best legal move
        const legalMoves = this.game.moves({ verbose: true });
        let bestMove = null;
        let bestScore = -Infinity;

        for (const move of legalMoves) {
            const moveIdx = this.moveToIndex(move);
            if (moveIdx !== null && policyLogits[moveIdx] > bestScore) {
                bestScore = policyLogits[moveIdx];
                bestMove = move;
            }
        }

        return bestMove;
    }

    encodeBoard() {
        // Encode board as 12 x 8 x 8 tensor (one plane per piece type per color)
        const board = new Float32Array(12 * 8 * 8);
        board.fill(0);

        const fen = this.game.fen().split(' ')[0];
        let square = 0;

        for (const char of fen) {
            if (char === '/') continue;

            if (char >= '1' && char <= '8') {
                square += parseInt(char);
            } else {
                const pieceIdx = this.pieceToIndex(char);
                if (pieceIdx !== null) {
                    board[pieceIdx * 64 + square] = 1;
                }
                square++;
            }
        }

        // Create tensor
        return new ort.Tensor('float32', board, [1, 12, 8, 8]);
    }

    getLegalMovesMask() {
        // Create mask for legal moves
        const mask = new Uint8Array(4672);
        mask.fill(0);

        const legalMoves = this.game.moves({ verbose: true });
        for (const move of legalMoves) {
            const idx = this.moveToIndex(move);
            if (idx !== null) {
                mask[idx] = 1;
            }
        }

        return new ort.Tensor('bool', mask, [1, 4672]);
    }

    pieceToIndex(piece) {
        const pieces = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        };
        return pieces[piece] ?? null;
    }

    moveToIndex(move) {
        // Simple move indexing (from_square * 64 + to_square)
        // This is a simplified version; your actual model might use a different encoding
        const fromIdx = this.squareToIndex(move.from);
        const toIdx = this.squareToIndex(move.to);

        if (fromIdx === null || toIdx === null) return null;

        // Simple encoding: from * 64 + to
        // For underpromotions, add offset (this is simplified)
        let idx = fromIdx * 64 + toIdx;

        // Handle promotions (simplified)
        if (move.promotion) {
            const promoOffset = {
                'q': 0, 'r': 4096, 'b': 4096 + 64, 'n': 4096 + 128
            };
            idx = 4096 + (promoOffset[move.promotion] || 0) + (fromIdx * 8 + (toIdx % 8));
        }

        return idx < 4672 ? idx : null;
    }

    squareToIndex(square) {
        if (square.length !== 2) return null;
        const file = square.charCodeAt(0) - 'a'.charCodeAt(0);
        const rank = parseInt(square[1]) - 1;
        if (file < 0 || file > 7 || rank < 0 || rank > 7) return null;
        return rank * 8 + file;
    }

    addMoveToHistory(move) {
        this.moveHistory.push(move);
        const moveNum = Math.ceil(this.moveHistory.length / 2);

        const moveList = document.getElementById('move-history');

        if (this.moveHistory.length % 2 === 1) {
            // White's move
            const div = document.createElement('div');
            div.className = 'move-item';
            div.innerHTML = `
                <span class="move-number">${moveNum}.</span>
                <span>${move.san}</span>
            `;
            moveList.appendChild(div);
        } else {
            // Black's move
            const lastMove = moveList.lastElementChild;
            lastMove.innerHTML += ` <span>${move.san}</span>`;
        }

        // Scroll to bottom
        moveList.scrollTop = moveList.scrollHeight;

        // Update move count
        document.getElementById('move-count').textContent = this.moveHistory.length;
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }

    updateTurnIndicator() {
        const turnDiv = document.getElementById('turn');
        const turn = this.game.turn();

        if (this.game.game_over()) {
            turnDiv.className = 'turn-indicator';
            turnDiv.textContent = '';
        } else if (turn === 'w') {
            turnDiv.className = 'turn-indicator turn-white';
            turnDiv.textContent = "White's turn";
        } else {
            turnDiv.className = 'turn-indicator turn-black';
            turnDiv.textContent = "Black's turn";
        }
    }

    checkGameOver() {
        if (this.game.game_over()) {
            let message = 'Game Over! ';

            if (this.game.in_checkmate()) {
                const winner = this.game.turn() === 'w' ? 'Black' : 'White';
                message += `${winner} wins by checkmate!`;
            } else if (this.game.in_stalemate()) {
                message += 'Draw by stalemate';
            } else if (this.game.in_threefold_repetition()) {
                message += 'Draw by repetition';
            } else if (this.game.insufficient_material()) {
                message += 'Draw by insufficient material';
            } else {
                message += 'Draw';
            }

            this.updateStatus(message);
            this.updateTurnIndicator();
            return true;
        }

        this.updateTurnIndicator();
        return false;
    }

    resetGame() {
        this.game.reset();
        this.board.start();
        this.moveHistory = [];
        document.getElementById('move-history').innerHTML = '';
        document.getElementById('move-count').textContent = '0';
        document.getElementById('think-time').textContent = '-';
        document.getElementById('evaluation').textContent = '0.0';

        this.updateStatus('Game reset! Make your move.');
        this.updateTurnIndicator();

        // If player is black, AI moves first
        if (this.playerColor === 'black') {
            setTimeout(() => this.makeAIMove(), 500);
        }
    }

    undoMove() {
        if (this.moveHistory.length < 2) {
            this.updateStatus('Cannot undo - not enough moves');
            return;
        }

        // Undo last 2 moves (player + AI)
        this.game.undo();
        this.game.undo();
        this.board.position(this.game.fen());

        // Remove from history
        this.moveHistory.pop();
        this.moveHistory.pop();

        // Update UI
        const moveList = document.getElementById('move-history');
        if (moveList.lastElementChild) {
            moveList.removeChild(moveList.lastElementChild);
        }

        document.getElementById('move-count').textContent = this.moveHistory.length;
        this.updateStatus('Moves undone. Your turn!');
        this.updateTurnIndicator();
    }
}

// Initialize the app when page loads
window.addEventListener('DOMContentLoaded', () => {
    window.chessAI = new ChessAI();
});
