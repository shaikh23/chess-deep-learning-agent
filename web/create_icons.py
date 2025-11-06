"""Create simple app icons for the Chess AI PWA.

This script creates basic icons. For better icons, use:
- https://www.favicon-generator.org/
- https://realfavicongenerator.net/
- Design your own in Photoshop/Figma/etc.
"""

try:
    from PIL import Image, ImageDraw, ImageFont
    import sys

    def create_icon(size, output_path):
        """Create a simple chess-themed icon."""
        # Create image with dark background
        img = Image.new('RGBA', (size, size), '#181818')
        draw = ImageDraw.Draw(img)

        # Draw a simple chess board pattern
        square_size = size // 8
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 0:
                    x = col * square_size
                    y = row * square_size
                    draw.rectangle(
                        [x, y, x + square_size, y + square_size],
                        fill='#4a90e2'
                    )

        # Add border
        border_width = size // 32
        draw.rectangle(
            [0, 0, size - 1, size - 1],
            outline='#4a90e2',
            width=border_width
        )

        # Try to add text
        try:
            font_size = size // 3
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            text = "♟"

            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Center text
            x = (size - text_width) // 2
            y = (size - text_height) // 2 - bbox[1]

            # Draw text with shadow for depth
            draw.text((x + 2, y + 2), text, font=font, fill='#000000')
            draw.text((x, y), text, font=font, fill='#ffffff')
        except:
            # If font fails, just use the checkerboard
            pass

        # Save
        img.save(output_path)
        print(f"Created {output_path} ({size}x{size})")

    print("Creating Chess AI app icons...")
    print()

    create_icon(192, 'icon-192.png')
    create_icon(512, 'icon-512.png')

    print()
    print("Icons created successfully!")
    print()
    print("For better custom icons, visit:")
    print("- https://www.favicon-generator.org/")
    print("- https://realfavicongenerator.net/")

except ImportError:
    print("Pillow not installed. Install with:")
    print("  pip install Pillow")
    print()
    print("Or create icons manually:")
    print("1. Create two PNG files: icon-192.png (192x192) and icon-512.png (512x512)")
    print("2. Use any chess-themed design")
    print("3. Or use online icon generator: https://www.favicon-generator.org/")
    sys.exit(1)
