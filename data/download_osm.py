import requests
import os

def download_osm_data(filename="data/coimbatore.osm"):
    # Coimbatore Bounding Box (approx)
    # South, West, North, East
    bbox = "10.99,76.93,11.03,76.98" 
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:xml][timeout:180];
    (
      way["highway"]({bbox});
      node(w);
    );
    out meta;
    >;
    out meta qt;
    """
    
    print(f"Downloading OSM data for bbox: {bbox}...")
    response = requests.get(overpass_url, params={'data': overpass_query})
    
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Successfully saved to {filename}")
    else:
        print(f"Error downloading data: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    download_osm_data()
