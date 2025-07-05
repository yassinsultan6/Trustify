# =============================================================================
# TRUSTIFY - AI Content Analysis Platform
# =============================================================================
# This Flask application provides AI-powered content analysis including:
# - Review authenticity detection
# - Spam message detection  
# - Fake news detection
# - Multi-review analysis with product comparison
# =============================================================================

# Flask web framework for creating the web application
from flask import Flask, request, render_template

# Python standard library imports
import pickle  # For loading pre-trained machine learning models
import re      # For regular expression operations in text processing
import traceback  # For error handling and debugging
import numpy   # For numerical operations

# Natural Language Processing libraries
import nltk    # Natural Language Toolkit for text processing
from nltk.corpus import stopwords      # Common words to filter out
from nltk.tokenize import word_tokenize  # Text tokenization
from nltk.stem import WordNetLemmatizer  # Word lemmatization

# Web scraping and HTTP requests
import requests  # HTTP library for making requests
from bs4 import BeautifulSoup  # HTML/XML parser for web scraping

# Machine Learning and AI libraries
from sklearn.feature_extraction.text import CountVectorizer  # Text feature extraction
from sentence_transformers import SentenceTransformer  # BERT-based text embeddings

# Web automation for dynamic content extraction
from playwright.sync_api import sync_playwright  # Browser automation for JavaScript-heavy sites

# =============================================================================
# FLASK APPLICATION INITIALIZATION
# =============================================================================

# Create Flask application instance
app = Flask(__name__)

# =============================================================================
# NLTK RESOURCE DOWNLOAD
# =============================================================================
# Download required NLTK data packages for text processing
# These are essential for tokenization, lemmatization, and stopword removal
nltk.download('punkt')      # Tokenizer data
nltk.download('wordnet')    # Lexical database for lemmatization
nltk.download('stopwords')  # Stopwords for various languages

# =============================================================================
# MACHINE LEARNING MODELS LOADING
# =============================================================================
# Load pre-trained models for different types of content analysis
# These models have been trained on large datasets for accuracy
try:
    # Review authenticity detection model (XGBoost-based)
    # This model determines if a product review is genuine or fake
    with open('xgboost_fake_news_model.pkl', 'rb') as f:
        review_model = pickle.load(f)

    # Spam detection model
    # Identifies spam messages and malicious content
    with open('spammodel1.pkl', 'rb') as f:
        spam_model = pickle.load(f)
 
    # Fake news detection model
    # Determines if news articles are authentic or fabricated
    with open('fakenewsmodel.pkl', 'rb') as f:
        fake_comment_model = pickle.load(f)

    # BERT (Bidirectional Encoder Representations from Transformers) transformer
    # Converts text into high-dimensional vectors for machine learning
    # Uses the 'all-MiniLM-L6-v2' model for efficient text embeddings
    bert_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    print("✅ Models and vectorizers loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# =============================================================================
# TEXT PREPROCESSING FUNCTIONS
# =============================================================================
# These functions clean and prepare text data for machine learning analysis
# Proper preprocessing is crucial for accurate AI predictions

def preprocess_text(text):
    """
    Main text preprocessing function that cleans and normalizes text.
    
    Args:
        text (str): Raw text input from user or web scraping
        
    Returns:
        str: Cleaned and normalized text ready for analysis
    """
    return clean_text(text)

def clean_text(text):
    """
    Performs comprehensive text cleaning operations.
    
    This function:
    1. Converts text to lowercase for consistency
    2. Removes special characters and punctuation
    3. Keeps only alphabetic characters and spaces
    
    Args:
        text (str): Raw text to be cleaned
        
    Returns:
        str: Cleaned text with only letters and spaces
    """
    # Convert to lowercase for consistent processing
    text = text.lower()
    
    # Remove all non-alphabetic characters except spaces
    # This helps focus on the actual content rather than formatting
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text

def get_platform_specific_selectors(url):
    """
    Returns platform-specific CSS selectors for better review extraction.
    
    This function analyzes the URL to determine which e-commerce platform
    the review is from and returns the most effective CSS selectors for
    that specific platform. This improves accuracy in review extraction.
    
    Different platforms structure their HTML differently, so using
    platform-specific selectors significantly improves success rates.
    
    Args:
        url (str): The URL of the page containing reviews
        
    Returns:
        list: List of CSS selectors optimized for the detected platform
    """
    # Convert URL to lowercase for case-insensitive matching
    url_lower = url.lower()
    
    # =============================================================================
    # AMAZON PLATFORM SELECTORS
    # =============================================================================
    # Amazon uses specific data attributes and classes for reviews
    if 'amazon' in url_lower:
        return [
            '[data-hook="review-body"]',    # Primary Amazon review container
            '.review-data',                 # Alternative review data class
            '.review-content',              # General review content
            '.a-expander-content',          # Expandable review content
            '[data-testid="review-body"]',  # Test ID for review body
            '.review-text',                 # Review text container
            '.review-comment'               # Review comment section
        ]
    
    # =============================================================================
    # YELP PLATFORM SELECTORS
    # =============================================================================
    # Yelp uses different class naming conventions for reviews
    elif 'yelp' in url_lower:
        return [
            '.review-content',              # Main review content
            '.review__content',             # Alternative review content
            '[data-review-id] p',           # Review paragraphs with ID
            '.review-text',                 # Review text
            '.comment'                      # Comment sections
        ]
    
    # =============================================================================
    # GOOGLE REVIEWS SELECTORS
    # =============================================================================
    # Google Reviews and Google Maps reviews
    elif 'google' in url_lower and ('review' in url_lower or 'maps' in url_lower):
        return [
            '.review-snippet',              # Review snippet container
            '.review-text',                 # Review text
            '.review-content',              # Review content
            '[data-review-id]',             # Review with ID
            '.comment'                      # Comment sections
        ]
    
    # =============================================================================
    # TRIPADVISOR PLATFORM SELECTORS
    # =============================================================================
    # TripAdvisor uses specific review containers
    elif 'tripadvisor' in url_lower:
        return [
            '.review-container',            # Review container
            '.review-content',              # Review content
            '.review-text',                 # Review text
            '.comment',                     # Comment sections
            '.review-body'                  # Review body
        ]
    
    # =============================================================================
    # TRUSTPILOT PLATFORM SELECTORS
    # =============================================================================
    # Trustpilot review structure
    elif 'trustpilot' in url_lower:
        return [
            '.review-content',              # Review content
            '.review-body',                 # Review body
            '.review-text',                 # Review text
            '.comment'                      # Comment sections
        ]
    
    # =============================================================================
    # GENERIC E-COMMERCE PLATFORMS
    # =============================================================================
    # Common selectors for major e-commerce sites
    elif any(site in url_lower for site in ['ebay', 'walmart', 'target', 'bestbuy', 'newegg']):
        return [
            '.review-content',              # Review content
            '.review-body',                 # Review body
            '.review-text',                 # Review text
            '.customer-review',             # Customer review
            '.product-review',              # Product review
            '.review-comment'               # Review comment
        ]
    
    # =============================================================================
    # DEFAULT/UNKNOWN PLATFORM SELECTORS
    # =============================================================================
    # Fallback selectors for unknown platforms
    # These are more generic but should work on most sites
    else:
        return [
            '[class*="review"]',            # Any class containing "review"
            '[id*="review"]',               # Any ID containing "review"
            '[data-testid*="review"]',      # Test ID containing "review"
            '.review-content',              # Review content
            '.review-body',                 # Review body
            '.review-text',                 # Review text
            '.comment',                     # Comment sections
            '.feedback'                     # Feedback sections
        ]

def fetch_dynamic_review(url):
    """
    Enhanced function to fetch review content from a given URL using Playwright.
    Specifically targets and extracts only the FIRST review from the page.
    
    This function uses Playwright (a browser automation tool) to handle
    JavaScript-heavy websites that traditional scraping methods can't process.
    It's particularly effective for modern e-commerce sites that load
    content dynamically.
    
    The function employs a multi-layered approach:
    1. Platform-specific CSS selectors for accuracy
    2. General review selectors as fallback
    3. Pattern matching for content validation
    4. Content quality checks to ensure meaningful reviews
    
    Args:
        url (str): The URL of the page containing the review
        
    Returns:
        str: The extracted review text, or None if no review found
    """
    print(f"DEBUG: Attempting to fetch FIRST review content from URL: {url} using Playwright.")
    
    try:
        # Initialize Playwright browser automation
        with sync_playwright() as p:
            # Launch a headless Chromium browser (no GUI)
            # Headless mode is faster and uses fewer resources
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Navigate to the target URL with error handling
            try:
                # Set a 60-second timeout for page loading
                page.goto(url, timeout=60000)
                # Wait 5 seconds for dynamic content to load
                page.wait_for_timeout(5000)
            except Exception as nav_e:
                print(f"DEBUG: Playwright navigation error for {url}: {nav_e}")
                browser.close()
                return None

            # Get the fully rendered HTML content after JavaScript execution
            content = page.content()
            # Parse the HTML using BeautifulSoup for easy element selection
            soup = BeautifulSoup(content, 'html.parser')

            # Get platform-specific CSS selectors for better accuracy
            platform_selectors = get_platform_specific_selectors(url)
            
            # Enhanced review-specific selectors as fallback
            # These are more generic but cover common review patterns
            general_selectors = [
                "review-content", "review-body", "full-review", "customer-review",
                "review-text", "review-description", "review-detail", "review-comment",
                "user-review", "product-review", "review-message", "review-summary",
                "[data-testid*='review']", "[class*='review']", "[id*='review']",
                "comment-content", "feedback-content", "opinion-content",
                "rating-content", "star-rating", "rating-text"
            ]
            
            # Combine platform-specific and general selectors for maximum coverage
            all_selectors = platform_selectors + general_selectors

            # Try each selector to find the FIRST review
            for selector in all_selectors:
                try:
                    # Find all elements matching the current selector
                    elements = soup.select(selector)
                    if elements:  # If we found elements with this selector
                        # Get the FIRST element only (we want the first review)
                        first_element = elements[0]
                        text = first_element.get_text(strip=True)
                        
                        # Check if the extracted text meets minimum quality criteria
                        if text and len(text) > 50:  # Minimum length for a meaningful review
                            # Validate that the text actually looks like a review
                            if is_review_like_content(text):
                                print(f"DEBUG: Found FIRST review content with selector '{selector}'. Length: {len(text)}")
                                browser.close()
                                return text
                except Exception as e:
                    print(f"DEBUG: Error with selector '{selector}': {e}")
                    continue

            # If no review found with CSS selectors, try pattern matching
            # This is a fallback method that looks for review-like text patterns
            extracted_text = find_first_review_content(soup)
            if extracted_text:
                print(f"DEBUG: Found FIRST review-like content using pattern matching. Length: {len(extracted_text)}")
                browser.close()
                return extracted_text

            browser.close()
            return None
    except Exception as e:
        print(f"❌ Playwright error in fetch_dynamic_review: {e}")
        return None

def is_review_like_content(text):
    """
    Validates if the extracted text appears to be review content.
    
    This function uses a scoring system to determine if the extracted text
    is likely to be a genuine product review. It looks for multiple indicators
    that are commonly found in review content.
    
    The validation process checks for:
    1. Review-specific keywords and phrases
    2. Rating patterns (stars, numbers, etc.)
    3. Personal pronouns indicating personal experience
    4. Text length and quality
    
    Args:
        text (str): The text to validate as review content
        
    Returns:
        bool: True if the text appears to be review content, False otherwise
    """
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # =============================================================================
    # REVIEW INDICATORS - KEYWORDS COMMONLY FOUND IN REVIEWS
    # =============================================================================
    # These words and phrases are typically present in genuine reviews
    review_indicators = [
        'review', 'rating', 'star', 'recommend', 'purchase', 'buy', 'product',
        'quality', 'experience', 'opinion', 'thought', 'feel', 'like', 'dislike',
        'good', 'bad', 'excellent', 'poor', 'amazing', 'terrible', 'worth',
        'customer', 'user', 'buyer', 'verified', 'purchased', 'bought'
    ]
    
    # Count how many review indicators are present in the text
    indicator_count = sum(1 for indicator in review_indicators if indicator in text_lower)
    
    # =============================================================================
    # RATING PATTERNS - COMMON RATING FORMATS
    # =============================================================================
    # Look for common rating patterns that indicate review content
    rating_patterns = [
        r'\d+\s*out\s*of\s*\d+',  # "5 out of 5"
        r'\d+\s*stars?',          # "5 stars"
        r'★+',                    # Star symbols
        r'rating\s*:\s*\d+',      # "rating: 5"
        r'\d+\s*/\s*\d+',         # "5/5"
    ]
    
    # Count how many rating patterns are found
    rating_matches = sum(1 for pattern in rating_patterns if re.search(pattern, text_lower))
    
    # =============================================================================
    # PERSONAL PRONOUNS - INDICATORS OF PERSONAL EXPERIENCE
    # =============================================================================
    # Personal pronouns suggest the text is written from personal experience
    personal_pronouns = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours']
    pronoun_count = sum(1 for pronoun in personal_pronouns if pronoun in text_lower.split())
    
    # =============================================================================
    # SCORING SYSTEM - COMBINE ALL INDICATORS
    # =============================================================================
    # Calculate a score based on all the indicators found
    score = 0
    
    # Add points for review indicators (2+ indicators = 2 points)
    if indicator_count >= 2:
        score += 2
    
    # Add points for rating patterns (1+ patterns = 2 points)
    if rating_matches >= 1:
        score += 2
    
    # Add points for personal pronouns (2+ pronouns = 1 point)
    if pronoun_count >= 2:
        score += 1
    
    # Add points for text length (longer text is more likely to be a review)
    if len(text) > 100:
        score += 1
    
    # Return True if the score meets the threshold (3+ points)
    return score >= 3

def find_first_review_content(soup):
    """
    Fallback method to find the FIRST review-like content using pattern matching.
    """
    # Look for paragraphs with review-like characteristics - get the FIRST one only
    elements = soup.find_all(['p', 'div', 'span'])
    
    for element in elements:
        text = element.get_text(strip=True)
        if text and len(text) > 100:  # Minimum length
            if is_review_like_content(text):
                return text  # Return the FIRST review found
    
    # If no review-like content found, return the FIRST meaningful text
    for element in soup.find_all(['p', 'div']):
        text = element.get_text(strip=True)
        if text and len(text) > 50 and not text.isdigit():
            return text  # Return the FIRST meaningful text found
    
    return None

def fetch_multiple_reviews(url, max_reviews=15):
    """
    Enhanced function to fetch multiple reviews (up to max_reviews) from a given URL using Playwright.
    Returns a list of review dictionaries with content and metadata.
    """
    print(f"DEBUG: Attempting to fetch up to {max_reviews} reviews from URL: {url} using Playwright.")
    reviews = []
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            try:
                page.goto(url, timeout=60000)
                page.wait_for_timeout(5000)
            except Exception as nav_e:
                print(f"DEBUG: Playwright navigation error for {url}: {nav_e}")
                browser.close()
                return reviews

            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')

            # Get platform-specific selectors
            platform_selectors = get_platform_specific_selectors(url)
            
            # Enhanced review-specific selectors (fallback)
            general_selectors = [
                "review-content", "review-body", "full-review", "customer-review",
                "review-text", "review-description", "review-detail", "review-comment",
                "user-review", "product-review", "review-message", "review-summary",
                "[data-testid*='review']", "[class*='review']", "[id*='review']",
                "comment-content", "feedback-content", "opinion-content",
                "rating-content", "star-rating", "rating-text"
            ]
            
            # Combine platform-specific and general selectors
            all_selectors = platform_selectors + general_selectors

            # Try each selector and find multiple reviews
            for selector in all_selectors:
                try:
                    elements = soup.select(selector)
                    if elements and len(reviews) < max_reviews:
                        for element in elements:
                            if len(reviews) >= max_reviews:
                                break
                                
                            text = element.get_text(strip=True)
                            
                            if text and len(text) > 50:  # Minimum length for a meaningful review
                                # Check if text contains review-like content
                                if is_review_like_content(text):
                                    # Try to extract rating if available
                                    rating = extract_rating_from_element(element)
                                    
                                    review_data = {
                                        'content': text,
                                        'rating': rating,
                                        'selector_used': selector,
                                        'length': len(text)
                                    }
                                    
                                    # Avoid duplicates
                                    if not any(r['content'] == text for r in reviews):
                                        reviews.append(review_data)
                                        print(f"DEBUG: Found review #{len(reviews)} with selector '{selector}'. Length: {len(text)}")
                                        
                except Exception as e:
                    print(f"DEBUG: Error with selector '{selector}': {e}")
                    continue

            # If we didn't find enough reviews with selectors, try pattern matching
            if len(reviews) < max_reviews:
                additional_reviews = find_multiple_review_content(soup, max_reviews - len(reviews))
                for review_text in additional_reviews:
                    if len(reviews) >= max_reviews:
                        break
                    
                    review_data = {
                        'content': review_text,
                        'rating': None,
                        'selector_used': 'pattern_matching',
                        'length': len(review_text)
                    }
                    
                    # Avoid duplicates
                    if not any(r['content'] == review_text for r in reviews):
                        reviews.append(review_data)
                        print(f"DEBUG: Found additional review #{len(reviews)} using pattern matching. Length: {len(review_text)}")

            browser.close()
            print(f"DEBUG: Total reviews found: {len(reviews)}")
            return reviews
            
    except Exception as e:
        print(f"❌ Playwright error in fetch_multiple_reviews: {e}")
        return reviews

def extract_rating_from_element(element):
    """
    Attempts to extract rating information from a review element.
    """
    try:
        # Look for rating in the element or its parent
        rating_selectors = [
            '.rating', '.stars', '.star-rating', '[data-rating]', '[class*="star"]',
            '[class*="rating"]', '.review-rating', '.rating-stars'
        ]
        
        # Check the element itself and its parent
        for check_element in [element, element.parent]:
            if check_element:
                for selector in rating_selectors:
                    rating_elem = check_element.select_one(selector)
                    if rating_elem:
                        rating_text = rating_elem.get_text(strip=True)
                        # Try to extract numeric rating
                        import re
                        rating_match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                        if rating_match:
                            return float(rating_match.group(1))
        
        return None
    except Exception as e:
        print(f"DEBUG: Error extracting rating: {e}")
        return None

def find_multiple_review_content(soup, max_reviews):
    """
    Fallback method to find multiple review-like content using pattern matching.
    """
    reviews = []
    elements = soup.find_all(['p', 'div', 'span'])
    
    for element in elements:
        if len(reviews) >= max_reviews:
            break
            
        text = element.get_text(strip=True)
        if text and len(text) > 100:  # Minimum length
            if is_review_like_content(text):
                # Avoid duplicates
                if text not in [r['content'] for r in reviews]:
                    reviews.append(text)
    
    # If no review-like content found, return meaningful text
    if not reviews:
        for element in soup.find_all(['p', 'div']):
            if len(reviews) >= max_reviews:
                break
                
            text = element.get_text(strip=True)
            if text and len(text) > 50 and not text.isdigit():
                # Avoid duplicates
                if text not in reviews:
                    reviews.append(text)
    
    return reviews

def analyze_multiple_reviews(reviews, item_details):
    """
    Analyzes multiple reviews and generates a comprehensive report.
    """
    report = {
        'total_reviews': len(reviews),
        'reviews_analyzed': [],
        'summary': {
            'real_count': 0,
            'fake_count': 0,
            'aligned_count': 0,
            'not_aligned_count': 0
        }
    }
    
    for i, review_data in enumerate(reviews):
        review_text = review_data['content']
        processed_text = preprocess_text(review_text)
        
        if not processed_text:
            continue
            
        # Analyze authenticity
        bert_features = bert_transformer.encode([processed_text])
        authenticity_prediction = review_model.predict(bert_features)[0]
        is_real = authenticity_prediction == 1
        
        # Analyze alignment with product description
        alignment_result = "Not applicable"
        common_keywords = []
        if item_details:
            item_keywords = extract_keywords(item_details)
            review_keywords = extract_keywords(processed_text)
            common_keywords = set(item_keywords) & set(review_keywords)
            
            if common_keywords:
                alignment_result = "Aligned"
            else:
                alignment_result = "Not aligned"
        
        # Update summary
        if is_real:
            report['summary']['real_count'] += 1
        else:
            report['summary']['fake_count'] += 1
            
        if alignment_result == "Aligned":
            report['summary']['aligned_count'] += 1
        elif alignment_result == "Not aligned":
            report['summary']['not_aligned_count'] += 1
        
        # Add review analysis to report
        review_analysis = {
            'review_number': i + 1,
            'content': review_text[:200] + "..." if len(review_text) > 200 else review_text,
            'full_content': review_text,
            'length': len(review_text),
            'rating': review_data.get('rating'),
            'authenticity': "Real" if is_real else "Fake",
            'alignment': alignment_result,
            'common_keywords': list(common_keywords) if item_details else []
        }
        
        report['reviews_analyzed'].append(review_analysis)
    
    return report

def get_platform_specific_product_selectors(url):
    """
    Returns platform-specific CSS selectors for better product description extraction.
    """
    url_lower = url.lower()
    
    # Amazon
    if 'amazon' in url_lower:
        return [
            '#productDescription',
            '#feature-bullets',
            '.product-description',
            '.a-expander-content',
            '[data-feature-name="productDescription"]',
            '.a-section.a-spacing-medium.a-spacing-top-small',
            '#productDetails_detailBullets_sections1',
            '#productDetails_techSpec_section_1',
            '.a-section.a-spacing-base',
            '.a-section.a-spacing-small'
        ]
    
    # eBay
    elif 'ebay' in url_lower:
        return [
            '.itemAttr',
            '.itemDescription',
            '.product-description',
            '.item-details',
            '.x-item-description',
            '.x-item-description__mainContent'
        ]
    
    # Walmart
    elif 'walmart' in url_lower:
        return [
            '.product-description',
            '.product-details',
            '.product-info',
            '.product-overview',
            '.product-specifications'
        ]
    
    # Target
    elif 'target' in url_lower:
        return [
            '.product-description',
            '.product-details',
            '.product-info',
            '.product-overview'
        ]
    
    # Best Buy
    elif 'bestbuy' in url_lower:
        return [
            '.product-description',
            '.product-details',
            '.product-info',
            '.product-overview',
            '.product-specifications'
        ]
    
    # Newegg
    elif 'newegg' in url_lower:
        return [
            '.product-description',
            '.product-details',
            '.product-info',
            '.product-overview',
            '.product-specifications'
        ]
    
    # Generic e-commerce sites
    else:
        return [
            '.product-description',
            '.product-details', 
            '.item-description',
            '.product-info',
            '.description',
            '.details',
            '.product-overview',
            '.product-specifications',
            '.item-info',
            '.product-summary'
        ]

def is_product_description_like_content(text):
    """
    Validates if extracted text appears to be product description content.
    """
    text_lower = text.lower()
    
    # Product description indicators
    product_indicators = [
        'product', 'item', 'description', 'features', 'specifications',
        'details', 'information', 'about', 'overview', 'summary',
        'dimensions', 'weight', 'material', 'color', 'size',
        'brand', 'model', 'type', 'category', 'manufacturer',
        'includes', 'package', 'contents', 'what\'s included',
        'technical', 'specs', 'characteristics', 'properties'
    ]
    
    # Technical specifications patterns
    spec_patterns = [
        r'\d+\s*(cm|inch|mm|kg|lb|g|oz)',  # Measurements
        r'\d+\s*(watt|volt|amp|hz|mhz|ghz)',  # Technical specs
        r'[a-z]+\s*:\s*[a-z0-9\s]+',  # Key-value pairs
        r'\d+\s*x\s*\d+',  # Dimensions like "10 x 5"
        r'\d+\s*%\s*(cotton|polyester|wool|silk)',  # Material percentages
    ]
    
    # Avoid review-related content
    review_indicators = [
        'review', 'rating', 'star', 'customer', 'buyer', 'verified',
        'purchased', 'bought', 'experience', 'opinion', 'thought'
    ]
    
    # Scoring system
    score = 0
    indicator_count = sum(1 for indicator in product_indicators if indicator in text_lower)
    spec_matches = sum(1 for pattern in spec_patterns if re.search(pattern, text_lower))
    review_count = sum(1 for indicator in review_indicators if indicator in text_lower)
    
    if indicator_count >= 3:
        score += 2
    if spec_matches >= 1:
        score += 2
    if len(text) > 200:  # Product descriptions are usually longer
        score += 1
    if review_count <= 1:  # Avoid review content
        score += 1
    
    return score >= 3

# Fetch item details from the URL - NOW USING PLAYWRIGHT
def fetch_item_details(url):
    """
    Enhanced function to fetch product description content using platform-specific selectors.
    """
    print(f"DEBUG: Attempting to fetch enhanced item details from URL: {url} using Playwright.")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            try:
                page.goto(url, timeout=60000)
                page.wait_for_timeout(5000) # Wait for page to load JS
            except Exception as nav_e:
                print(f"DEBUG: Playwright navigation error for item details {url}: {nav_e}")
                browser.close()
                return None

            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')

            # Get platform-specific selectors
            platform_selectors = get_platform_specific_product_selectors(url)
            
            # Try platform-specific selectors first
            for selector in platform_selectors:
                try:
                    elements = soup.select(selector)
                    if elements:
                        for element in elements:
                            text = element.get_text(strip=True)
                            if text and len(text) > 100:
                                if is_product_description_like_content(text):
                                    print(f"DEBUG: Found product description with platform selector '{selector}'. Length: {len(text)}")
                                    browser.close()
                                    return text
                except Exception as e:
                    print(f"DEBUG: Error with platform selector '{selector}': {e}")
                    continue

            # Fallback to generic selectors
            generic_selectors = [
                '.product-description', '.description', '.details',
                '.product-info', '.item-info', '.product-details',
                '.product-overview', '.product-specifications',
                '.item-description', '.product-summary'
            ]
            
            for selector in generic_selectors:
                try:
                    elements = soup.select(selector)
                    if elements:
                        for element in elements:
                            text = element.get_text(strip=True)
                            if text and len(text) > 100:
                                if is_product_description_like_content(text):
                                    print(f"DEBUG: Found product description with generic selector '{selector}'. Length: {len(text)}")
                                    browser.close()
                                    return text
                except Exception as e:
                    print(f"DEBUG: Error with generic selector '{selector}': {e}")
                    continue

            # Final fallback: look for any meaningful content that might be product description
            print("DEBUG: No product description found with selectors, trying fallback method...")
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text and len(text) > 200:  # Longer text is more likely to be description
                    if is_product_description_like_content(text):
                        print(f"DEBUG: Found product description in paragraph fallback. Length: {len(text)}")
                        browser.close()
                        return text

            browser.close()
            print("DEBUG: No product description content found.")
            return None
            
    except Exception as e:
        print(f"❌ Playwright error in fetch_item_details: {e}")
        return None

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_keywords(text, percentage=0.1):
    """
    Extracts the most important keywords from text using TF-IDF vectorization.
    
    This function identifies the most significant words in the text that
    can be used for comparison and analysis. It's particularly useful for
    comparing review content with product descriptions.
    
    Args:
        text (str): The text to extract keywords from
        percentage (float): Percentage of words to extract (default: 0.1 = 10%)
        
    Returns:
        list: List of extracted keywords
    """
    try:
        if not text:
            return []
           
        # Split text into words
        words = text.split()
        
        # Calculate number of keywords to extract (minimum 5, maximum based on percentage)
        num_keywords = max(5, int(len(words) * percentage))
        
        # Use CountVectorizer to extract keywords
        # This removes stopwords and focuses on meaningful terms
        vectorizer = CountVectorizer(stop_words='english', max_features=num_keywords)
        vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out()
        
        return keywords
    except Exception as e:
        print(f"❌ Error extracting keywords: {e}")
        return []

# =============================================================================
# FLASK ROUTES - WEB INTERFACE
# =============================================================================

# Basic page routes for navigation
@app.route("/")
def landing():
    """Landing page route."""
    return render_template("landing.html")

@app.route("/intro")
def intro():
    """Introduction page route."""
    return render_template("intro.html")

@app.route("/signup")
def signup():
    """Signup page route."""
    return render_template("signup.html")

# =============================================================================
# MAIN REVIEW ANALYSIS ROUTE
# =============================================================================
@app.route('/analyze_review', methods=['GET', 'POST'])
def analyze_review():
    """
    Main route for analyzing review authenticity and comparing with product details.
    
    This is the core functionality of the Trustify platform. It provides:
    1. Review authenticity detection using AI
    2. Product description extraction and comparison
    3. Multiple review analysis when available
    4. Keyword-based alignment checking
    
    The function supports two input methods:
    - Text input: Direct text submission
    - URL input: Automatic content extraction from e-commerce pages
    
    Args:
        POST request with either 'review_text' or 'link' parameter
        
    Returns:
        Rendered result page with analysis results
    """
    if request.method == 'POST':
        # Get the input type (text or URL)
        input_type = request.form['input_type']
        
        # Initialize result variables
        comparison_result = None      # Product comparison result
        text = ""                     # Review text to analyze
        multiple_reviews_report = None # Multiple reviews analysis report

        # =============================================================================
        # INPUT PROCESSING - HANDLE TEXT VS URL INPUT
        # =============================================================================
        if input_type == 'text':
            # Direct text input from user
            text = request.form['review_text']
            item_details = None  # No product details for text input
            print("DEBUG: Input type is 'text'.")
            
        elif input_type == 'link':
            # URL input - extract content from web page
            link = request.form['link']
            print(f"DEBUG: Input type is 'link'. URL: {link}")
            
            # =============================================================================
            # MULTIPLE REVIEWS ANALYSIS
            # =============================================================================
            # First, try to fetch multiple reviews for comprehensive analysis
            multiple_reviews = fetch_multiple_reviews(link)
            print(f"DEBUG: Multiple reviews found: {len(multiple_reviews)}")
            
            if len(multiple_reviews) > 1:
                # Multiple reviews found - perform comprehensive analysis
                print(f"DEBUG: Analyzing {len(multiple_reviews)} reviews...")
                
                # Fetch product details for comparison
                item_details = fetch_item_details(link)
                print(f"DEBUG: Item details fetched (length): {len(item_details) if item_details else 0}")
                
                # Analyze all reviews and generate comprehensive report
                multiple_reviews_report = analyze_multiple_reviews(multiple_reviews, item_details)
                
                # Use the first review for the main analysis (backward compatibility)
                text = multiple_reviews[0]['content']
                print(f"DEBUG: Using first review for main analysis. Length: {len(text)}")
            else:
                # Single review - use dynamic review extraction
                text = fetch_dynamic_review(link) 
                print(f"DEBUG: Single review fetched from URL (length): {len(text) if text else 0}")

            # =============================================================================
            # ERROR HANDLING - CHECK FOR VALID CONTENT
            # =============================================================================
            if not text:
                print("DEBUG: 'text' is empty or None after fetch_dynamic_review.")
                return render_template('result.html', 
                                     result="Please submit your review as a text or check the URL.", 
                                     review_text="", 
                                     comparison_result=None, 
                                     multiple_reviews_report=None)

            # Fetch product details for comparison (only for URL input)
            item_details = fetch_item_details(link) 
            print(f"DEBUG: Item details fetched (length): {len(item_details) if item_details else 0}")

        # =============================================================================
        # TEXT PREPROCESSING
        # =============================================================================
        # Clean and normalize the text for AI analysis
        processed_text = preprocess_text(text)
        print(f"DEBUG: Processed text length: {len(processed_text) if processed_text else 0}")
        
        if not processed_text:
            print("DEBUG: Processed text is empty or None.")
            return render_template('result.html', 
                                 result="❌ Text processing failed.", 
                                 review_text="", 
                                 comparison_result=None, 
                                 multiple_reviews_report=None)

        # =============================================================================
        # PRODUCT COMPARISON ANALYSIS
        # =============================================================================
        # Compare review content with product description (only for URL input)
        if input_type == 'link' and item_details:
            # Extract keywords from both review and product description
            item_keywords = extract_keywords(item_details)
            review_keywords = extract_keywords(processed_text)

            # Find common keywords between review and product
            common_keywords = set(item_keywords) & set(review_keywords)
            print(f"DEBUG: Common keywords: {common_keywords}")
            
            # Determine if review aligns with product description
            if common_keywords:
                comparison_result = "✅ Review aligns with product description."
            else:
                comparison_result = "⚠️ No match found between review and description."
        elif input_type == 'link' and not item_details:
            comparison_result = "⚠️ Item description not found, skipping comparison."

        # =============================================================================
        # AI-BASED AUTHENTICITY ANALYSIS
        # =============================================================================
        # Use BERT transformer to convert text to high-dimensional vectors
        bert_features = bert_transformer.encode([processed_text])
        
        # Use the trained review model to predict authenticity
        prediction = review_model.predict(bert_features)[0]
        
        # Generate human-readable result
        result_text = "✅ Real Review" if prediction == 1 else "This review seems to be fake or computer generated"
        print(f"DEBUG: Prediction result: {result_text}")

        # =============================================================================
        # RETURN RESULTS
        # =============================================================================
        # Render the result page with all analysis data
        return render_template('result.html', 
                             result=result_text, 
                             review_text=text[:500],  # Show first 500 chars
                             comparison_result=comparison_result, 
                             multiple_reviews_report=multiple_reviews_report)

    # GET request - show the analysis form
    return render_template("analyze_review.html", analysis_type="Review")

# =============================================================================
# SPAM ANALYSIS ROUTE
# =============================================================================
@app.route('/analyze_spam', methods=['GET', 'POST'])
def analyze_spam():
    """
    Route for analyzing text to detect spam and malicious content.
    
    This function uses a pre-trained machine learning model to identify
    spam messages, phishing attempts, and other malicious content.
    
    Args:
        POST request with 'review_text' parameter
        
    Returns:
        Rendered result page with spam detection results
    """
    if request.method == 'POST':
        # Get text input from form
        text = request.form.get('review_text')  # Only supports text input
        
        # Validate input
        if not text:
            return render_template('result.html', result="❌ No text submitted.", review_text="")

        # Preprocess the text for analysis
        processed_text = preprocess_text(text)
        if not processed_text:
            return render_template('result.html', result="❌ Text processing failed.", review_text="")

        # =============================================================================
        # AI-BASED SPAM DETECTION
        # =============================================================================
        # Convert text to BERT embeddings for analysis
        bert_features = bert_transformer.encode([processed_text])
        
        # Use the trained spam detection model
        prediction = spam_model.predict(bert_features)[0]
        
        # Generate result message
        result_text = "❌ Spam Detected" if prediction == 1 else "✅ Safe Message"

        return render_template('result.html', result=result_text, review_text=text[:500])

    # GET request - show the spam analysis form
    return render_template("spam_analyze.html", analysis_type="Spam")

# =============================================================================
# FAKE NEWS ANALYSIS ROUTE
# =============================================================================
@app.route('/analyze_fake_comment', methods=['GET', 'POST'])
def analyze_comment():
    """
    Route for analyzing news articles and comments for authenticity.
    
    This function uses AI to determine if news content is authentic or
    fabricated. It's particularly useful for detecting fake news and
    misinformation.
    
    Args:
        POST request with 'review_text' parameter
        
    Returns:
        Rendered result page with fake news detection results
    """
    if request.method == 'POST':
        # Get text input from form
        text = request.form.get('review_text')  # Only supports text input
        
        # Validate input
        if not text:
            return render_template('result.html', result="❌ No text submitted.", review_text="")

        # Preprocess the text for analysis
        processed_text = preprocess_text(text)
        if not processed_text:
            return render_template('result.html', result="❌ Text processing failed.", review_text="")

        # =============================================================================
        # AI-BASED FAKE NEWS DETECTION
        # =============================================================================
        # Convert text to BERT embeddings for analysis
        bert_features = bert_transformer.encode([processed_text])
        
        # Use the trained fake news detection model
        prediction = fake_comment_model.predict(bert_features)[0]
        
        # Generate result message
        result_text = "✅ These news seems to be true" if prediction == 1 else "❌ These news seems to be fake or computer generated"

        return render_template('result.html', result=result_text, review_text=text[:500])

    # GET request - show the fake news analysis form
    return render_template("analyze_fake_comment.html", analysis_type="Spam")

# =============================================================================
# ERROR HANDLERS
# =============================================================================
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors - page not found."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """
    Handle 500 errors - internal server errors.
    
    This function logs the full error traceback for debugging purposes
    and returns a user-friendly error message.
    """
    # Log the full traceback for debugging purposes
    traceback.print_exc()
    return f"❌ Internal Server Error: {e}", 500

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    # Run the Flask application in debug mode on port 5002
    # Debug mode enables detailed error messages and auto-reload
    app.run(debug=True, port=5002)