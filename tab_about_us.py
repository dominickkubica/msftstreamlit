import streamlit as st
from pathlib import Path
import base64
from PIL import Image, ImageDraw

###############################################################################
#  Photo location helpers
###############################################################################
BASE_DIR = Path(__file__).resolve().parent
# Create a specific assets directory for images
ASSETS_DIR = BASE_DIR / "assets"

# Ensure the assets directory exists
ASSETS_DIR.mkdir(exist_ok=True)

# Use exact filenames from the repository as shown in the screenshot
PHOTO_MAP = {
    "Dylan Gordon":       "DYLAN.png",
    "Dominick Kubica":    "DOMINICK.png",
    "Nanami Emura":       "NANAMI.png",
    "Derleen Saini":      "DERLEEN.png",
    "Charles Goldenberg": "charles.png",  # Lowercase as seen in the screenshot
}

def get_image_as_base64(file_path):
    """Convert an image file to base64 for reliable display"""
    try:
        with open(file_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        return encoded
    except Exception as e:
        # Use print instead of st.debug
        print(f"Error encoding image {file_path}: {e}")
        return None

def display_profile_image(name: str):
    """
    Display a teammate's photo or a gray placeholder if the file is missing.
    Uses consistent sizing for all images.
    """
    # 1) Use exact filename from PHOTO_MAP 
    filename = PHOTO_MAP.get(name)

    # Try multiple locations for the image file
    possible_paths = [
        BASE_DIR / filename,    # Check main directory first (as shown in screenshot)
        ASSETS_DIR / filename,  # Then check assets directory
        Path(filename)          # Finally check relative path
    ]
    
    img_found = False
    for img_path in possible_paths:
        if img_path.exists():
            # Use base64 encoding for reliable image display
            encoded_image = get_image_as_base64(img_path)
            if encoded_image:
                # Set consistent image styling with fixed height and width
                html = f'''
                <div class="image-container">
                    <img src="data:image/png;base64,{encoded_image}" 
                    class="profile-image">
                </div>
                '''
                st.markdown(html, unsafe_allow_html=True)
                img_found = True
                break
    
    if not img_found:
        # Create a nicer placeholder with person icon
        size = 200
        ph = Image.new("RGB", (size, size), color="#E0E0E0")
        d = ImageDraw.Draw(ph)
        d.text((size * 0.25, size * 0.45), f"No Photo\n{name.split()[0]}", fill="#555555")
        
        # Display placeholder with consistent container sizing
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(ph, width=200)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Log the error
        print(f"No image found for {name}. Checked paths: {possible_paths}")

###############################################################################
#  Main component
###############################################################################
def render_about_us_tab(microsoft_colors: dict):
    st.markdown("## Meet Our Team")
    st.markdown(
        "We are a group of passionate data scientists and financial analysts "
        "working to revolutionize how earnings calls are analyzed."
    )

    # Add CSS for better styling
    st.markdown("""
    <style>
    .profile-card {
        background-color: #f9f9f9;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        height: 100%;
        display: flex;
        flex-direction: column;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .profile-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .profile-card h3 {
        margin-top: 12px;
        margin-bottom: 5px;
        color: #2F2F2F;
    }
    .profile-content {
        flex-grow: 1;
    }
    .image-container {
        height: 200px;
        display: flex;
        justify-content: center;
        align-items: center;
        overflow: hidden;
        border-radius: 8px;
    }
    .profile-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-radius: 8px;
        transition: transform 0.3s;
    }
    .profile-image:hover {
        transform: scale(1.03);
    }
    </style>
    """, unsafe_allow_html=True)

    team_members = [
        # (unchanged) ----------------------------------------------------------
        {
            "name": "Dylan Gordon",
            "role": "University Researcher",
            "about": "Dylan is a former chemical engineer turned data scientist. "
                     "He specializes in optimization and machine learning.",
            "interests": "Stocks, Pickleball, Boxing",
            "contact": "dtgordon@scu.edu",
        },
        {
            "name": "Dominick Kubica",
            "role": "University Researcher",
            "about": "Dominick is an aspiring home-grown data scientist with a "
                     "passion for finance and technology. ML and AI enthusiast.",
            "interests": "Data Science, Weightlifting, Cooking",
            "contact": "dominickkubica@gmail.com",
        },
        {
            "name": "Nanami Emura",
            "role": "University Researcher",
            "about": "Nanami developed the core sentiment analysis algorithm "
                     "and specializes in transformer models for financial text analysis.",
            "interests": "Deep Learning, Soccer, Photography",
            "contact": "nemura@scu.edu",
        },
        {
            "name": "Derleen Saini",
            "role": "University Researcher",
            "about": "Derleen created this Streamlit application and specializes in "
                     "data visualization and user-experience design.",
            "interests": "UI/UX Design, Photography, Yoga",
            "contact": "dsaini@scu.edu",
        },
        {
            "name": "Charles Goldenberg",
            "role": "Practicum Project Lead",
            "about": "Charles is the leader of our practicum project and has extensive "
                     "experience working with technical projects.",
            "interests": "Statistical Modeling, Travel, Jazz",
            "contact": "cgoldenberg@scu.edu",
        },
    ]
    
    # Improved layout with even spacing - first row of 3, second row of 2 centered
    col1, col2, col3 = st.columns(3)
    col4, empty_col, col5 = st.columns([1, 1, 1])  # Center the bottom row
    
    columns = [col1, col2, col3, col4, col5]
    
    for i, member in enumerate(team_members):
        with columns[i]:
            st.markdown('<div class="profile-card">', unsafe_allow_html=True)
            
            display_profile_image(member["name"])

            st.markdown(
                f"""
                <div class="profile-content">
                    <h3>{member['name']}</h3>
                    <p style="color:{microsoft_colors['primary']};font-weight:bold;">
                        {member['role']}
                    </p>
                    <p>{member['about']}</p>
                    <p><strong>Interests:</strong> {member['interests']}</p>
                    <p><strong>Contact:</strong> {member['contact']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

###############################################################################
#  Quick standalone test
###############################################################################
if __name__ == "__main__":
    st.set_page_config(page_title="About Us Demo", layout="wide")
    
    # Display setup instructions for first-time users
    if not (Path(__file__).parent / "assets").exists():
        st.warning("""
        **Image Setup Required**: 
        
        Please place your team member images in the same directory as this script:
        - DYLAN.png
        - DOMINICK.png
        - NANAMI.png
        - DERLEEN.png 
        - charles.png (lowercase as shown in repository)
        """)
    
    render_about_us_tab({"primary": "#0078d4"})