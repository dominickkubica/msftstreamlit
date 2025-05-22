import streamlit as st
from pathlib import Path
import base64
from PIL import Image, ImageDraw
import io

###############################################################################
#  Photo location helpers
###############################################################################

PHOTO_MAP = {
    "Dylan Gordon":       "DJR38489.JPG",
    "Dominick Kubica":    "DOMINICK.jpg",
    "Nanami Emura":       "NANAMI.jpg",
    "Derleen Saini":      "DJR38151.JPG",
    "Charles Goldenberg": "Goldenberg_Charles_1.jpg",
}

def optimize_image_for_web(image_path, target_width=300, target_height=300, quality=85):
    """
    Optimize image for web display with consistent sizing and quality
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate dimensions to maintain aspect ratio while fitting in target size
            img_ratio = img.width / img.height
            target_ratio = target_width / target_height
            
            if img_ratio > target_ratio:
                # Image is wider - fit to height and center crop width
                new_height = target_height
                new_width = int(target_height * img_ratio)
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Center crop
                left = (new_width - target_width) // 2
                img_cropped = img_resized.crop((left, 0, left + target_width, target_height))
            else:
                # Image is taller - fit to width and center crop height
                new_width = target_width
                new_height = int(target_width / img_ratio)
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Center crop
                top = (new_height - target_height) // 2
                img_cropped = img_resized.crop((0, top, target_width, top + target_height))
            
            # Convert to base64 with optimized quality
            buffer = io.BytesIO()
            img_cropped.save(buffer, format='JPEG', quality=quality, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode()
            return encoded
            
    except Exception as e:
        print(f"Error optimizing image {image_path}: {e}")
        return None

def create_placeholder_image(size=300, name="No Photo"):
    """Create a high-quality placeholder image"""
    img = Image.new("RGB", (size, size), color="#f0f2f6")
    draw = ImageDraw.Draw(img)
    
    # Draw a subtle border
    draw.rectangle([0, 0, size-1, size-1], outline="#e1e5eb", width=2)
    
    # Add icon-like placeholder
    center = size // 2
    draw.ellipse([center-40, center-60, center+40, center+20], fill="#cbd5e0")
    draw.rectangle([center-50, center+10, center+50, center+50], fill="#cbd5e0")
    
    # Add text
    try:
        # Try to use a better font size
        from PIL import ImageFont
        font = ImageFont.load_default()
        text = name.split()[0] if name != "No Photo" else "No Photo"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text(((size - text_width) // 2, center + 60), text, fill="#4a5568", font=font)
    except:
        # Fallback for basic text
        draw.text((center - 30, center + 60), name[:8], fill="#4a5568")
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return encoded

def find_profile_image(name: str):
    """
    Find and return optimized base64 image for a team member
    """
    # 1) Exact mapping first
    filename = PHOTO_MAP.get(name)

    # 2) If not in map: try FIRSTNAME.png (legacy convenience)
    if filename is None:
        filename = f"{name.split()[0].upper()}.png"

    # Try multiple locations for the image file
    script_dir = Path(__file__).parent
    
    possible_paths = [
        script_dir / filename,
        script_dir.parent / filename,
        Path(filename),
        script_dir / "assets" / filename,
        script_dir / filename.lower(),
        script_dir.parent / filename.lower(),
        Path(filename.lower()),
    ]
    
    for img_path in possible_paths:
        if img_path.exists():
            optimized_image = optimize_image_for_web(img_path)
            if optimized_image:
                print(f"Successfully loaded and optimized image for {name} from: {img_path}")
                return optimized_image
    
    # If no image found, create placeholder
    print(f"No image found for {name}. Creating placeholder.")
    return create_placeholder_image(name=name)

###############################################################################
#  Main component
###############################################################################
def render_about_us_tab(microsoft_colors: dict):
    # Enhanced CSS for responsive design
    st.markdown("""
    <style>
    /* Container for the entire about section */
    .about-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    /* Team grid container */
    .team-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 2rem;
        margin-top: 2rem;
    }
    
    /* Individual profile cards */
    .profile-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 
            0 8px 25px rgba(0,0,0,0.08),
            0 2px 10px rgba(0,0,0,0.04);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(0,0,0,0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
        position: relative;
        overflow: hidden;
    }
    
    .profile-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #0078d4, #106ebe);
    }
    
    .profile-card:hover {
        transform: translateY(-8px);
        box-shadow: 
            0 20px 40px rgba(0,0,0,0.12),
            0 8px 20px rgba(0,0,0,0.08);
    }
    
    /* Profile image container */
    .profile-image-container {
        width: 100%;
        display: flex;
        justify-content: center;
        margin-bottom: 1.5rem;
    }
    
    .profile-image {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid #ffffff;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .profile-card:hover .profile-image {
        transform: scale(1.05);
    }
    
    /* Profile content */
    .profile-content {
        text-align: center;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    
    .profile-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0 0 0.5rem 0;
        line-height: 1.2;
    }
    
    .profile-role {
        color: #0078d4;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .profile-about {
        color: #4a5568;
        line-height: 1.6;
        margin-bottom: 1.5rem;
        flex-grow: 1;
        font-size: 0.95rem;
        text-align: center;
    }
    
    .profile-details {
        border-top: 1px solid #e2e8f0;
        padding-top: 1.5rem;
        margin-top: auto;
        text-align: center;
    }
    
    .profile-interests, .profile-contact {
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
    }
    
    .profile-interests strong, .profile-contact strong {
        color: #2d3748;
        font-weight: 600;
    }
    
    .profile-interests span, .profile-contact span {
        color: #4a5568;
    }
    
    .profile-contact a {
        color: #0078d4;
        text-decoration: none;
        transition: color 0.2s ease;
    }
    
    .profile-contact a:hover {
        color: #106ebe;
        text-decoration: underline;
    }
    
    /* Header styling */
    .team-header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .team-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #0078d4, #106ebe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .team-description {
        font-size: 1.1rem;
        color: #4a5568;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .about-container {
            padding: 1rem 0.5rem;
        }
        
        .team-grid {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        
        .profile-card {
            padding: 1.5rem;
        }
        
        .profile-image {
            width: 150px;
            height: 150px;
        }
        
        .team-title {
            font-size: 2rem;
        }
        
        .team-description {
            font-size: 1rem;
        }
    }
    
    @media (max-width: 480px) {
        .profile-image {
            width: 120px;
            height: 120px;
        }
        
        .profile-name {
            font-size: 1.3rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Team data
    team_members = [
        {
            "name": "Dominick Kubica",
            "role": "University Researcher", 
            "about": "Dominick is an aspiring home-grown data scientist with a passion for finance and technology. ML and AI enthusiast focused on practical applications.",
            "interests": "Data Science, Weightlifting, Cooking",
            "contact": "dominickkubica@gmail.com",
        },
        {
            "name": "Dylan Gordon",
            "role": "University Researcher",
            "about": "Dylan is a former chemical engineer turned data scientist. He specializes in optimization and machine learning, bringing analytical precision to financial data analysis.",
            "interests": "Stocks, Pickleball, Boxing",
            "contact": "dtgordon@scu.edu",
        },
        {
            "name": "Nanami Emura",
            "role": "University Researcher",
            "about": "Nanami developed the core sentiment analysis algorithm and specializes in transformer models for financial text analysis, pushing the boundaries of NLP in finance.",
            "interests": "Deep Learning, Soccer, Photography", 
            "contact": "nemura@scu.edu",
        },
        {
            "name": "Derleen Saini",
            "role": "University Researcher",
            "about": "Derleen created this Streamlit application and specializes in data visualization and user-experience design, making complex data accessible and beautiful.",
            "interests": "UI/UX Design, Photography, Yoga",
            "contact": "dsaini@scu.edu",
        },
        {
            "name": "Charles Goldenberg", 
            "role": "Practicum Project Lead",
            "about": "Charles is the leader of our practicum project and has extensive experience working with technical projects, guiding the team toward innovative solutions.",
            "interests": "Statistical Modeling, Travel, Jazz",
            "contact": "cgoldenberg@scu.edu",
        },
    ]

    # Render the page
    st.markdown('<div class="about-container">', unsafe_allow_html=True)
    
    # Header section
    st.markdown("""
    <div class="team-header">
        <h1 class="team-title">Meet Our Team</h1>
        <p class="team-description">
            We are a group of passionate data scientists and financial analysts 
            working to revolutionize how earnings calls are analyzed through 
            cutting-edge machine learning and intuitive design.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Team grid
    st.markdown('<div class="team-grid">', unsafe_allow_html=True)
    
    for member in team_members:
        # Get optimized image
        profile_image_b64 = find_profile_image(member["name"])
        
        # Create profile card
        card_html = f"""
        <div class="profile-card">
            <div class="profile-image-container">
                <img src="data:image/jpeg;base64,{profile_image_b64}" 
                     class="profile-image" 
                     alt="{member['name']}" />
            </div>
            <div class="profile-content">
                <h2 class="profile-name">{member['name']}</h2>
                <p class="profile-role">{member['role']}</p>
                <p class="profile-about">{member['about']}</p>
                <div class="profile-details">
                    <p class="profile-interests">
                        <strong>Interests:</strong> <span>{member['interests']}</span>
                    </p>
                    <p class="profile-contact">
                        <strong>Contact:</strong> 
                        <span><a href="mailto:{member['contact']}">{member['contact']}</a></span>
                    </p>
                </div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close team-grid
    st.markdown('</div>', unsafe_allow_html=True)  # Close about-container

###############################################################################
#  Quick standalone test
###############################################################################
if __name__ == "__main__":
    st.set_page_config(page_title="About Us Demo", layout="wide")
    
    # Display setup instructions for first-time users
    if not any(Path(__file__).parent.glob("*.jpg")) and not any(Path(__file__).parent.glob("*.png")):
        st.info("""
        **📸 Image Setup**: 
        
        Place your team member images in the same directory as this script or in an 'assets' folder.
        The app will automatically find and optimize them for the best display quality.
        
        **Supported formats:** JPG, PNG, JPEG
        """)
    
    render_about_us_tab({"primary": "#0078d4"})
