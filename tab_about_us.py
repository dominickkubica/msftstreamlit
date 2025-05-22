import streamlit as st
from pathlib import Path
import base64
from PIL import Image, ImageDraw

###############################################################################
#  Photo location helpers
###############################################################################

PHOTO_MAP = {
    "Dylan Gordon":       "DYLAN.jpg",
    "Dominick Kubica":    "DOMINICK.jpg",
    "Nanami Emura":       "NANAMI.jpg",
    "Derleen Saini":      "DERLEEN.jpg",
    "Charles Goldenberg": "charles.jpg",
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
    # 1) Exact mapping first
    filename = PHOTO_MAP.get(name)

    # 2) If not in map: try FIRSTNAME.png (legacy convenience)
    if filename is None:
        filename = f"{name.split()[0].upper()}.png"

    # Try multiple locations for the image file
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    possible_paths = [
        # Check in the same directory as the script
        script_dir / filename,
        # Check in the root directory (one level up if script is in a subdirectory)
        script_dir.parent / filename,
        # Check relative to current working directory
        Path(filename),
        # Check in an assets folder
        script_dir / "assets" / filename,
        # Check for common variations (lowercase)
        script_dir / filename.lower(),
        script_dir.parent / filename.lower(),
        # Check in root with lowercase
        Path(filename.lower()),
    ]
    
    img_found = False
    for img_path in possible_paths:
        if img_path.exists():
            try:
                # Use base64 encoding for reliable image display
                encoded_image = get_image_as_base64(img_path)
                if encoded_image:
                    # Set consistent image styling with fixed height and width
                    html = f'''
                    <div style="height:200px; display:flex; justify-content:center; align-items:center; overflow:hidden;">
                        <img src="data:image/png;base64,{encoded_image}" 
                        style="width:100%; height:200px; object-fit:cover; border-radius:8px;">
                    </div>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                    img_found = True
                    print(f"Successfully loaded image for {name} from: {img_path}")
                    break
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
    
    if not img_found:
        # Draw simple placeholder with consistent sizing
        size = 200
        ph = Image.new("RGB", (size, size), color="#CCCCCC")
        d = ImageDraw.Draw(ph)
        d.text((size * 0.35, size * 0.45), "No\nPhoto", fill="black")
        
        # Display placeholder with consistent container sizing
        st.markdown(
            '''
            <div style="height:200px; display:flex; justify-content:center; align-items:center;">
            ''', 
            unsafe_allow_html=True
        )
        st.image(ph, width=200)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced debugging information
        print(f"No image found for {name}. Filename: {filename}")
        print(f"Searched paths:")
        for path in possible_paths:
            print(f"  - {path} (exists: {path.exists()})")
        print(f"Current working directory: {Path.cwd()}")
        print(f"Script directory: {script_dir}")

###############################################################################
#  Main component
###############################################################################
def render_about_us_tab(microsoft_colors: dict):
    st.markdown("## Meet Our Team")
    st.markdown(
        "We are a group of passionate data scientists and financial analysts "
        "working to revolutionize how earnings calls are analyzed."
    )

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

    # Add CSS for profile cards
    st.markdown("""
    <style>
    .profile-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .profile-card h3 {
        margin-top: 10px;
        margin-bottom: 5px;
    }
    .profile-content {
        flex-grow: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    # layout: 3 + 2
    rows = [st.columns(3), st.columns(2)]
    idx = 0
    for cols in rows:
        for col in cols:
            if idx >= len(team_members):
                break
            member = team_members[idx]
            idx += 1

            with col:
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

    # Contact form removed as requested

###############################################################################
#  Quick standalone test
###############################################################################
if __name__ == "__main__":
    st.set_page_config(page_title="About Us Demo", layout="wide")
    
    # Display setup instructions for first-time users
    if not (Path(__file__).parent / "assets").exists():
        st.warning("""
        **Image Setup Required**: 
        
        Please create an 'assets' folder in the same directory as this script and place your team member images there:
        - DYLAN.png
        - DOMINICK.png
        - NANAMI.png
        - DERLEEN.png
        - charles.png
        """)
    
    render_about_us_tab({"primary": "#0078d4"})
