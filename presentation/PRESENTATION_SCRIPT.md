# Presentation Script: Chess AI Project
## 10-Minute Technical Presentation with Simple Explanations

**Total Time: ~10 minutes**

---

## Slide 1: Title (30 seconds)

**[Show title slide]**

Hello everyone! Today I'm excited to share a project where I taught a computer to play chess by learning from master players. This isn't just about chess—it's about showing how modern AI can solve complex problems affordably and practically.

Over the next 10 minutes, I'll walk you through the problem I tackled, how I used machine learning to solve it, and the results I achieved.

---

## Slide 2: The Problem (1 minute)

**[Advance to problem slide]**

So let me start with the challenge. You've probably heard of chess engines like AlphaZero that can beat the best humans in the world. These systems are incredible, but they have a big problem: they're extremely expensive to build.

AlphaZero required **weeks of training** on **thousands of specialized processors** that cost over **$100,000** to run. For most people—students, researchers, hobbyists—this is completely out of reach.

So I asked myself: **Can we build something that's still smart and competitive, but doesn't break the bank?**

Specifically, I wanted to create a chess AI that:
- Trains in under an hour on a single graphics card
- Reaches intermediate club-level strength (around 2000 Elo—that's a solid amateur player)
- Costs less than 10 dollars to train
- Learns from expert human games instead of expensive self-play

The answer turned out to be **yes**—and I'll show you how.

---

## Slide 3: Why Chess? (45 seconds)

**[Advance to "Why Chess?" slide]**

Before diving into the solution, you might wonder: why chess?

Chess is actually a **perfect testing ground** for AI because:
- Every game has a **clear outcome**—win, loss, or draw—which gives perfect feedback
- Despite simple rules, chess has enormous complexity: about **10 to the power of 43** legal positions
- It requires **pattern recognition**, not just brute-force calculation
- We have **strong benchmarks** like Stockfish and human ratings to measure performance

But most importantly: we have **fantastic data**. The Lichess website offers a free database with **3.1 gigabytes** of games played by expert players—all available for anyone to use. That's millions of positions where masters showed us the right moves to make.

This combination of complexity, clear goals, and rich data makes chess ideal for machine learning.

---

## Slide 4: Our Approach (1 minute)

**[Advance to approach comparison slide]**

Now let's talk about the approach. There are two main ways to build a chess AI, and they involve very different trade-offs.

**The AlphaZero Approach** uses something called **reinforcement learning**. The computer plays millions of games against itself, gradually discovering what works. This is incredibly powerful—it can reach superhuman levels—but it requires weeks of training on massive hardware costing over $100,000.

**My Approach** uses **supervised learning**. Instead of learning from scratch by playing itself, the computer learns by **studying millions of moves from expert human games**. It's like learning chess by watching masters play and imitating their decisions.

Here's the comparison:
- **Training time**: 40 minutes versus weeks
- **Hardware**: One GPU versus thousands of processors
- **Cost**: $10 versus $100,000
- **Final strength**: About 2000 Elo (intermediate) versus 3000+ Elo (superhuman)

For practical applications where you don't need to beat Magnus Carlsen, supervised learning offers the **best bang for your buck**.

---

## Slide 5: The Data (1 minute)

**[Advance to data slide, gesture to phase distribution chart]**

So let's talk about the data—the foundation of any machine learning project.

I started with the **Lichess Elite Database**: 3.1 gigabytes of game files. But I couldn't use everything—I needed quality over quantity.

I filtered for games where **both players had ratings above 2000**. These are strong club players and masters—people who know what they're doing. From these games, I extracted **3.7 million positions**, each one labeled with what move the master played.

Now, I did some smart preprocessing:
- I **skipped the first 8 moves** of each game. Why? Because those early moves are often memorized opening theory, not real thinking.
- I **balanced the dataset** to have equal positions from white's and black's perspective
- I **tagged positions by game phase**: opening, middlegame, or endgame—this helps the AI learn different stages have different priorities

Finally, I split the data: **70% for training**, 15% for validation, and 15% for testing.

**[Gesture to chart]** This chart shows the distribution across game phases—you can see we have a good balance, with most positions coming from the complex middlegame where the real chess happens.

---

## Slide 6: The Neural Network (1.5 minutes)

**[Advance to neural network architecture slide]**

Now for the brain of the operation: the **neural network**.

I designed something called a **Mini-ResNet**, inspired by AlphaZero but scaled down to be more efficient. Let me break down what this means in simple terms.

**The Input**: The network sees the chess board as a **12 by 8 by 8 tensor**. Think of it like a stack of 12 images of the board, where each image highlights one type of piece—white pawns, black knights, white rooks, and so on. This gives the network a structured way to "see" the board.

**The Processing**: The board goes through what we call **6 residual blocks**. These are layers of the network that learn to recognize patterns—things like "this pawn structure is weak" or "the queen and rook are aligned for a tactic." The "residual" part is a clever trick that helps the network learn deeper patterns without getting confused.

**The Output**: Here's where it gets interesting. The network has **two heads**—two separate outputs:

1. **The Policy Head** answers: "What move should I play?" It gives a probability for each of the 4,672 possible moves. The network is essentially saying, "I think this move is good, this one is okay, this one is bad."

2. **The Value Head** answers: "Who's winning?" It outputs a single number from -1 to +1, where +1 means "I'm completely winning," 0 means "equal position," and -1 means "I'm losing badly."

The whole network has **6 million parameters**—the knobs it can tune during learning—and fits in a **23-megabyte file**. For comparison, AlphaZero uses over 100 million parameters.

One crucial feature: **legal move masking**. Before the network makes predictions, we tell it which moves are actually legal. This prevents it from ever suggesting something impossible like moving a pawn backwards.

---

## Slide 7: Training Setup (1 minute)

**[Advance to training slide, gesture to training curves]**

Now let's talk about how we actually teach this network.

**The Setup**:
- I used a learning algorithm called **AdamW**—think of it as a smart way to adjust those 6 million parameters
- **Batch size of 256** means the network looks at 256 positions at once before updating
- I trained for **10 epochs**—that means it saw the entire 3.7 million position dataset 10 times
- Hardware: **Google Colab** with an L4 GPU (anyone can access this for free or cheap)
- Total time: **40 minutes**

**The Loss Function**: This is how we measure mistakes. I combined two types of errors:
- **Policy Loss**: How often does the network pick the wrong move?
- **Value Loss**: How accurate is its evaluation of who's winning?

The formula weighs policy mistakes more heavily because picking good moves is slightly more important than perfect evaluation.

**[Gesture to chart]** This chart shows the training curves. The blue line is training performance, orange is validation. You can see both losses going down steadily, then plateauing around epoch 8 or 9—that's when the network has learned as much as it can from this data. The gap between training and validation is small, which means we're not overfitting—the network generalizes well to new positions.

---

## Slide 8: Training Results (1 minute)

**[Advance to results slide]**

So how well did it learn? Let's look at the numbers.

The network achieved **28.8% top-1 accuracy**. That means in about **1 out of every 3 positions**, the network's first choice **exactly matches** what the master played.

Now, you might think 28.8% sounds low, but context matters. Remember, there are about **35 legal moves** on average in a chess position. If you guessed randomly, you'd only be right **2.9% of the time**. So 28.8% is **10 times better than random**!

Even more impressive:
- **52% top-3 accuracy**: Over half the time, the master's move is in the network's top 3 predictions
- **64% top-5 accuracy**: Nearly two-thirds of the time, it's in the top 5

For comparison, human masters around 2200 rating are estimated to achieve about 40-50% top-1 accuracy. So our network is playing at a level between strong club players and masters—exactly what we aimed for.

**Two Key Insights**:

1. **Data matters more than architecture**. When I scaled from 100,000 positions to 3.7 million—37 times more data—accuracy jumped by 44%. That was way more impactful than any architectural tweaks I tried.

2. **Supervised learning is efficient**. Just 40 minutes of training on a single GPU gave the network intermediate chess understanding. That's the power of learning from experts.

---

## Slide 9: The Search Algorithm (1 minute)

**[Advance to search slide]**

Now here's the thing: the neural network alone isn't enough to play good chess. It can tell you "this move looks good," but chess requires **looking ahead**—thinking about what your opponent might do, what you'd do next, and so on.

This is where we combine neural AI with **classical search algorithms**. Specifically, I used something called **alpha-beta search**, enhanced with neural network guidance.

**Here's how it works, step by step**:

1. **Generate moves**: Get all legal moves in the current position
2. **Neural ordering**: Use the policy network to sort these moves—try the most promising ones first
3. **Search the tree**: For each move, imagine making it, then the opponent's response, then your next move, and so on. We look ahead 2-3 moves (called "plies") into the future.
4. **Prune bad branches**: This is the clever part of alpha-beta. If we discover a move is bad, we can skip exploring it further, saving tons of computation.
5. **Leaf evaluation**: When we reach the end of our search, use the value network to judge who's winning
6. **Back up scores**: Propagate those evaluations back up the tree
7. **Choose the best**: Pick the move that leads to the best outcome

**Optimizations**: I added several tricks to make this faster:
- **Transposition table**: Remember positions we've seen before so we don't recalculate them
- **Killer moves**: Remember moves that worked well recently and try them first
- **Quiescence search**: When there are captures or checks, look a bit deeper to avoid being fooled

With a **300-millisecond time limit** per move, the search typically reaches **depth 2-3** and evaluates a few hundred positions. That might sound shallow, but the neural network's quality evaluation compensates—depth 3 with smart evaluation often beats depth 5 with simple material counting.

---

## Slide 10: Benchmark Results (1 minute)

**[Advance to benchmark results slide]**

Alright, the moment of truth: **how strong is this AI**?

I ran tournaments against three different opponents, playing 100 games against each.

**Versus Sunfish (a simple Python engine around 2000 Elo)**:
- Record: **45 wins, 36 draws, 19 losses**
- Win rate: **63%**
- **We're stronger by about 92 Elo points**

This is great! A high draw rate shows solid positional understanding, and a win-loss ratio of more than 2-to-1 shows tactical competence.

**Versus Stockfish Level 1** (the world's strongest engine, but limited):
- Record: **12 wins, 6 draws, 82 losses**
- Win rate: **15%**
- We're losing by about 301 Elo

This was expected. Stockfish searches 10-15 moves deep compared to our 2-3. But we still won 12 games! The pattern is usually: we play competitively for 20-30 moves, then Stockfish finds a deep tactic we missed.

**Versus Maia-1500** (another neural engine trained to play like humans):
- Record: **1 win, 2 draws, 97 losses**
- This was disappointing, but interesting. Maia was trained on different data with a different goal (imitating human mistakes), so there's a distribution mismatch.

**Bottom line**: Our estimated Elo is around **2000**, which is **intermediate club level**—Class A to Expert in the US Chess Federation rating system. That's exactly what we aimed for!

---

## Slide 11: Performance Visualizations (30 seconds)

**[Advance to visualization slide, gesture to charts]**

These charts visualize our performance.

**[Point to left chart]** On the left, you see our relative Elo compared to each opponent—we're clearly beating Sunfish, losing to Stockfish and Maia.

**[Point to right chart]** On the right is "Average Centipawn Loss" by game phase. Think of this as average mistake size, where lower is better. You can see we make fewer mistakes than Sunfish across all phases—opening, middlegame, and endgame. Our opening play is particularly strong because we learned from master games.

---

## Slide 12: What Did It Learn? (45 seconds)

**[Advance to "What Did It Learn" slide]**

So what did the AI actually understand about chess?

**Strengths**:
- **Positional understanding**: It knows to control the center, develop pieces to good squares
- **Basic tactics**: It recognizes forks, pins, discovered attacks—the bread and butter of chess tactics
- **Opening principles**: The first 10-15 moves are strong, following patterns from the training data
- **Pattern recognition**: It sees common motifs like weak pawn structures or piece coordination

**Weaknesses**:
- **Deep tactics** (42% of errors): It misses combinations that require looking 4 or more moves ahead
- **Endgame precision** (28%): In technical endgames, it doesn't always know the optimal conversion technique
- **Positional nuances** (18%): It sometimes misses subtle prophylactic moves—moves that prevent opponent plans
- **King safety** (12%): Occasionally undervalues defensive moves

**Playing Style**: It plays like an **intermediate club player**—strong fundamentals and pattern recognition, but occasionally blunders complex tactics that require deep calculation.

---

## Slide 13: Key Insights (45 seconds)

**[Advance to key insights slide]**

Let me share five key takeaways from this project:

**1. Supervised Learning Is Practical**
- 40 minutes and $10 achieved real chess strength. You don't need DeepMind's budget to build useful AI!

**2. Data Quality Matters Most**
- 37× more data gave 44% better accuracy. This beat any architectural improvement I tried. The lesson: focus on data first.

**3. Hybrid Approaches Work Best**
- Neural networks provide pattern recognition, classical algorithms provide logical search. Together, they're better than either alone, especially under resource constraints.

**4. Trade-offs Are Real**
- Supervised learning is fast, cheap, simple → reaches ~2000 Elo
- Reinforcement learning is slow, expensive, complex → reaches 3000+ Elo
- Choose based on your goals and resources!

**5. AI Is Increasingly Accessible**
- What required institutional resources in 2010 now takes one person and cloud GPUs. AI democratization is real and accelerating.

---

## Slide 14: Future Improvements (30 seconds)

**[Advance to future improvements slide]**

This project could be extended in many ways. Let me highlight a few:

**Quick wins** with high return on investment:
- **Opening book**: Pre-compute optimal openings for +50-100 Elo (2 days of work)
- **Endgame tablebases**: Perfect 6-piece endgame play for +30-50 Elo (1 day)
- **Model quantization**: Make the network 2-3× faster so we can search deeper

**Medium-term improvements**:
- Scale to 10 million positions for +100-150 Elo
- Train a larger 20 million parameter model for +80-120 Elo
- Implement parallel search across multiple CPU cores

With all these improvements, the engine could potentially reach **2300-2400 Elo**—Expert to Master level.

---

## Slide 15: System Architecture (30 seconds)

**[Advance to architecture slide]**

Let me quickly summarize the complete system architecture.

**Training phase**:
- Start with 3.1GB of chess games
- Filter and parse into 3.7 million positions
- Feed through the MiniResNet neural network
- Optimize with the loss function
- 40 minutes later: trained model

**Inference phase** (when actually playing):
- Take current board position
- Encode as a 12×8×8 tensor
- Neural network outputs policy and value
- Alpha-beta search uses these to explore the game tree
- Choose best move and repeat

**Benchmarking**:
- Play 100 games against each opponent
- Collect statistics
- Estimate Elo rating

It's a complete machine learning pipeline from raw data to competitive gameplay.

---

## Slide 16: Project Statistics (30 seconds)

**[Advance to statistics slide]**

Just to give you a sense of scope, this project involved:
- **8,781 lines of Python code** across 33 modules
- **5 Jupyter notebooks** documenting the workflow
- **3.7 million training positions** from 3.1GB of data
- A **6 million parameter neural network**
- **300 benchmark games** against different opponents

Training took **40 minutes** and cost about **$10** in cloud GPU time.

The result: an engine rated around **2000 Elo**, capable of playing at intermediate club level.

It's a complete demonstration of the full machine learning pipeline, from data collection through training to evaluation and benchmarking.

---

## Slide 17: Conclusion (45 seconds)

**[Advance to conclusion slide]**

Let me wrap up.

**The problem**: Build competitive chess AI affordably, without massive institutional resources.

**The approach**: Supervised learning from 3.7 million master-level positions.

**The method**: Mini-ResNet neural network combined with alpha-beta search.

**The result**: An engine rated ~2000 Elo, trained in 40 minutes for $10.

**What we demonstrated**:
- Deep learning fundamentals applied to a complex problem
- The complete ML pipeline from data to deployment
- How to combine neural and classical AI
- Working within practical resource constraints

**Broader impact**:
- AI doesn't require massive budgets anymore
- Supervised learning is practical for real applications
- Understanding trade-offs helps choose the right approach
- In machine learning, data quality often matters more than architectural sophistication

This project shows that anyone with curiosity, a laptop, and access to cloud GPUs can build intelligent systems that would have been cutting-edge research a decade ago. **AI is accessible to everyone!**

---

## Slide 18: Thank You (15 seconds)

**[Advance to final slide]**

Thank you for your attention! I'm happy to answer any questions about the project—whether it's about the neural network architecture, the training process, the search algorithm, or anything else.

**[Pause for questions]**

---

## Tips for Delivery

### Timing Control
- **Total presentation time**: ~10 minutes
- Use a timer during practice
- If running over, shorten Slides 12-14 (insights, future work, architecture)
- If running under, expand on Slides 6 (neural network) or 9 (search)

### Body Language
- **Gesture to slides** when referencing specific charts or numbers
- **Make eye contact** with different parts of the audience
- **Move naturally**—don't stand frozen at one spot
- **Smile and show enthusiasm**—you're excited about this project!

### Vocal Variety
- **Vary your pace**: Slow down for complex concepts (neural network, search), speed up for familiar material
- **Emphasize key numbers**: "10 times better," "40 minutes," "2000 Elo"
- **Pause after important points** to let them sink in
- **Use inflection** to show excitement or importance

### Handling Technical Details
- **Assume a general technical audience**—they understand programming but maybe not deep learning
- **Use analogies**: Neural network "sees" the board, search "thinks ahead"
- **Define jargon first time**: "Elo rating—a number that measures chess strength"
- **Show, don't just tell**: Point to visualizations when discussing results

### Common Questions to Prepare For

**Q: Why not use reinforcement learning?**
A: Great question! RL can achieve higher strength, but it requires weeks of training and massive compute resources. For this project, I wanted to show that supervised learning offers excellent cost-performance trade-off. If you have $100K and a month of time, RL is amazing—but most people don't.

**Q: Could this approach work for other games?**
A: Absolutely! Any game with rich expert data and clear outcomes is a candidate. Go, poker, StarCraft—supervised learning from expert demonstrations is a powerful general approach.

**Q: What was the hardest part?**
A: Two things: (1) Processing 3.1GB of chess data efficiently without running out of memory, which is why I used the sharding approach, and (2) Debugging the search algorithm—making sure it was both correct and efficient required careful optimization.

**Q: How does this compare to Chess.com or Lichess engines?**
A: Online chess sites typically use Stockfish, which is much stronger—around 3500+ Elo. My engine at 2000 Elo is more like an intermediate club player. But my engine is pure machine learning, while Stockfish uses decades of hand-crafted evaluation—different philosophies!

**Q: Could you beat this engine?**
A: [Be honest about your own chess level!] If you're above 2000 Elo, you'd likely win most games. If you're below, it would be competitive. The engine plays like a solid Class A player—strong fundamentals but occasionally blunders tactics.

**Q: What would you do differently next time?**
A: I'd prioritize scaling data earlier—going from 100K to 3.7M positions was the biggest performance jump. I'd also implement opening books from the start, and consider INT8 quantization to speed up inference for deeper search.

### Demo Considerations

If you want to show a **live demo** (add 2-3 minutes to presentation):
- Have a saved position ready (avoid starting from move 1)
- Show 2-3 moves maximum
- Narrate what the engine is thinking
- Have backup screenshots in case technical difficulties arise

Example demo script:
> "Let me quickly show you the engine in action. Here's a position from the middlegame. The engine is thinking... and it plays Knight to f5, forking the queen and threatening the king. This is exactly the kind of tactical pattern it learned from the training data. Notice it only took 300 milliseconds to find this move."

### Closing Strong

End with confidence and openness:
- Smile
- Thank the audience sincerely
- Invite questions enthusiastically
- If no questions immediately: "While you think of questions, I'm also happy to discuss specific technical details like the residual blocks or move encoding."

### Practice Recommendations

1. **Practice 3-5 times out loud** before the presentation
2. **Record yourself** and watch for filler words ("um," "uh," "like")
3. **Time each section** to know where you can speed up or slow down
4. **Practice with a friend** and get feedback
5. **Stand up while practicing**—it's different from sitting!

---

## Simplified Analogies for Complex Concepts

If audience seems confused, use these analogies:

**Neural Network**:
> "Think of the neural network like a chess student who has watched thousands of master games. They've internalized patterns—they 'just know' when a position looks good, even if they can't explain exactly why. That's what the network learns."

**Residual Blocks**:
> "Residual connections are like having a mentor who shows you not just what changed, but also what stayed the same. This helps the network learn deeper patterns without getting confused."

**Alpha-Beta Pruning**:
> "Imagine you're at a restaurant looking at the menu. You see the first steak is $50, so you know you won't order any steak over $50. Alpha-beta pruning works the same way—once you find a good move, you can ignore branches that are clearly worse without fully exploring them."

**Top-1 Accuracy**:
> "If I show you a chess position and ask 'what would a master play?', you might not get it exactly right, but you might be close. Top-1 accuracy means the network's first guess exactly matches the master. Top-3 means the master's move is in the network's top 3 guesses."

**Supervised vs Reinforcement Learning**:
> "Supervised learning is like learning to drive with an instructor—you watch what they do and copy it. Reinforcement learning is like learning to drive by trial and error in an empty parking lot—you crash a lot at first, but eventually figure it out. The instructor method is faster, but trial-and-error might discover techniques the instructor never taught."

---

Good luck with your presentation! You've built an impressive, complete machine learning system, and this script will help you communicate its value clearly.