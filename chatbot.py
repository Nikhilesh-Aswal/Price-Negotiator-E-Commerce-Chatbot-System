import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

model = joblib.load('model/model_svm.joblib')  
scaler = joblib.load('model/scaler.joblib') 

data_path = 'data/flipkart_products_updated.csv'
data = pd.read_csv(data_path)

app = Flask(__name__)

categories = data['product_category'].unique()

selected_category = None
matching_products = []

def get_discount_prediction(retail_price):
    """Get the discount prediction from the SVM model"""
    scaled_price = scaler.transform([[retail_price]]) 
    discount = model.predict(scaled_price)[0]  
    return discount

@app.route("/", methods=["GET", "POST"])
def index():
    global selected_category, matching_products 

    if request.method == "POST":
        user_input = request.json.get("user_input").lower()
        print("User input:", user_input)  

        
        if user_input in ["hello", "hi", "hey"]:
            return jsonify({"message": "Hello! How can I assist you today?"})

       
        elif "looking for" in user_input:
            categories_list = ", ".join(categories)
            print("Categories available:", categories_list)  
            return jsonify({"message": f"Sure! Which category are you interested in? Here are some options: {categories_list}."})

       
        elif any(category.lower() in user_input for category in categories):
            selected_category = next(category for category in categories if category.lower() in user_input)
            print("Selected category:", selected_category)  

           
            products_in_category = data[data['product_category'] == selected_category]['product_name'].unique()[:10]
            print("Products in selected category (max 10):", products_in_category) 

            products_list = "\n".join(f"{i+1}. {product}" for i, product in enumerate(products_in_category))
            return jsonify({"message": f"Here are some products in the {selected_category} category:\n{products_list}\nPlease choose a product by name or number."})

       
        elif selected_category:
           
            matching_products = data[(data['product_category'] == selected_category) & (data['product_name'].str.lower() == user_input)].head(10).to_dict('records')
            print("Matching products:", matching_products)  

            if len(matching_products) == 1:
            
                product_info = matching_products[0]
                price = product_info['retail_price']
                print("Selected product:", product_info['product_name'], "Price:", price) 
                return jsonify({"message": f"The price of {product_info['product_name']} is ${price}. Would you like to 'buy' or 'negotiate'?"})
            elif len(matching_products) > 1:
            
                product_variants = "\n".join(f"{i+1}. {p['product_name']} - ${p['retail_price']}" for i, p in enumerate(matching_products))
                return jsonify({"message": f"There are multiple products with the name '{user_input}'. Please choose the one you are interested in by specifying the number:\n{product_variants}"})
            else:
                return jsonify({"message": f"I'm sorry, I couldn't find that product in the {selected_category} category. Please check the name or try another product."})

       
        elif len(matching_products) > 1 and user_input.isdigit():
        
            index = int(user_input) - 1
            if 0 <= index < len(matching_products):
                selected_product = matching_products[index]
                price = selected_product['retail_price']
                print("Confirmed selected product:", selected_product['product_name'], "Price:", price)  # Debug
                return jsonify({"message": f"The price of {selected_product['product_name']} is ${price}. Would you like to 'buy' or 'negotiate'?"})
            else:
                return jsonify({"message": "Invalid selection. Please try again with a valid number."})

       
        elif "negotiate" in user_input and matching_products:
            retail_price = matching_products[0]['retail_price']
            discount = get_discount_prediction(retail_price)
            negotiated_price = retail_price - (retail_price * discount / 100) 
            print("Negotiated price:", negotiated_price) 
            return jsonify({"message": f"Negotiation successful! The new price is ${negotiated_price:.2f}. Would you like to 'buy' or 'not'?"})

        
        elif "buy" in user_input:
            print("User chose to buy:", matching_products[0]['product_name'])  
            return jsonify({"message": f"Thank you for your purchase of {matching_products[0]['product_name']}! Your order has been placed."})

        elif "not" in user_input:
            print("User chose not to buy.")
            return jsonify({"message": "No problem! Let me know if there's anything else I can help you with."})

       
        else:
            print("Unrecognized input:", user_input)  
            return jsonify({"message": "I didn't quite understand. Can you please clarify?"})

    
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
