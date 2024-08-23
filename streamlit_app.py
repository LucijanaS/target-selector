import json
import csv
from brightstar_functions import *
import streamlit as st
import pandas as pd
from orbit import Orbit
from correl import visib
from fourier import grids
from graphics import draw
import plotly.express as px
from plotly.subplots import make_subplots
import datetime




# - * - coding: utf - 8 - * -

# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------

lambda_U = 364 * 10 ** (-9)
lambda_V = 540 * 10 ** (-9)
lambda_B = 442 * 10 ** (-9)

n_brightest_stars = 10000

# Load the JSON data
with open('bsc5-all.json', 'r') as file:
    stars_data = json.load(file)

# Filter out stars with empty BayerF and Common column
stars_data = [star for star in stars_data if (star.get('BayerF') != '' and star.get('BayerF') is not None) or
                                              (star.get('Common') != '' and star.get('Common') is not None)]
# Sort stars_data based on Vmag (magnitude) in descending order
brightest_stars = sorted(stars_data, key=lambda x: float(x['Vmag']))[:n_brightest_stars]

# Process stars data
data = [process_star(star) for star in brightest_stars]

# Write data to CSV file
csv_columns = ["BayerF", "Common", "Parallax", "Distance", "Umag", "Vmag", "Bmag", "Temp", "RA_decimal", "Dec_decimal",
               "RA", "Dec", "Diameter_U", "Diameter_V", "Diameter_B", "Phi_U", "Phi_V", "Phi_B"]
csv_file = str(n_brightest_stars)+"stars_data.csv"
with open(csv_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(data)



df_all_stars = pd.read_csv(csv_file)

# Exclude the last 6 columns
df_display = df_all_stars.iloc[:, :-6]

# Display the DataFrame in Streamlit
st.write(df_display)




# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------



st.markdown(
    """
    ## Identifying SII Candidates by Location and Date
    On this webpage, you can find stars that are ideal candidates for stellar intensity interferometry. \n

    By entering your location and date, you can either view a list of available stars visible from that location or select a specific star 
    from the dropdown menu to track its path across the sky and explore its visibility map."""
    )

st.markdown(
    """
    Similarly to the Target Stars WebApp (https://target-stars-sii.streamlit.app/), the stars used are from the Yale Bright Star Catalog which contains 9110 of the brightest stars (found at http://tdc-www.harvard.edu/catalogs/bsc5.html). 
    The cataloge is in ASCII format and was converted to a .JSON format in the repository https://github.com/brettonw/YaleBrightStarCatalog. 
    The data file used from that repository is bsc5-all.json. \n
    From those 9000 stars, only the brightest 1500 were used for this WebApp of which a smaller set can be used to limit the runtime for the search.
    """
    )

st.markdown(
    """
    ### Enter parameters
    Before you proceed, it is best to enter all the parameters you have, like the coordinates of the location, the baseline, the time of observation and so on. \n
    You can do so in the sidebar on the left. Note that everytime you change something in the sidebar, the plots and tables will reload f.
    """
    )


# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------


st.sidebar.markdown("## Select Date and Location of Observation")



date = st.sidebar.date_input("Enter date of observation", datetime.date.today())


coordinates_form = st.sidebar.radio(
    "Enter coordinates in: ",
    ["degrees, minute, second (DMS)", "decimal degrees (DD)"])

two_telescopes = st.sidebar.radio(
    label="Two telescopes?",
    help="If you want to check the trace on the visibility map, you need to select two telescopes and enter at least the baseline. Otherwise you can still search for the stars visible from your location.",
    options=["Yes", "No"]
)
if two_telescopes=="Yes":
    two_telescopes_location = st.sidebar.radio(
        "Enter second telescope as baseline or coordinate",
        ["baseline", "coordinates"])


if coordinates_form == "degrees, minute, second (DMS)":
    if two_telescopes == "No":

        degree_lat1 = st.sidebar.number_input("Latitude degree:", format="%1f", value=23.)
        min_lat1 = st.sidebar.number_input("Latitude minute:", format="%1f", value=20.)
        sec_lat1 = st.sidebar.number_input("Latitude second:", format="%.1f", value=31.9)
        NS_lat1 = st.sidebar.selectbox("North or South:", ("N", "S"))

        degree_lon1 = st.sidebar.number_input("Longitude degree:", format="%1f", value=16.)
        min_lon1 = st.sidebar.number_input("Longitude minute:", format="%1f", value=13.)
        sec_lon1 = st.sidebar.number_input("Longitude second:", format="%.1f", value=29.7)
        WE_lon1 = st.sidebar.selectbox("West or East:", ("W", "E"))
    
        lat_deg1 = str(degree_lat1) + " " + str(min_lat1) + " " + str(sec_lat1) + NS_lat1
        lon_deg1 = str(degree_lon1) + " " + str(min_lon1) + " " + str(sec_lon1) + WE_lon1

        lat_dec1 = dms_to_decimal(lat_deg1)
        lon_dec1 = dms_to_decimal(lon_deg1)

    if two_telescopes == "Yes":
    
        degree_lat1 = st.sidebar.number_input("Latitude degree of the first telescope:", format="%1f", value=23.)
        min_lat1 = st.sidebar.number_input("Latitude minute of the first telescope:", format="%1f", value=20.)
        sec_lat1 = st.sidebar.number_input("Latitude second of the first telescope:", format="%.1f", value=31.9)
        NS_lat1 = st.sidebar.selectbox("North or South:", ("N", "S"), index=1)

        degree_lon1 = st.sidebar.number_input("Longitude degree of the first telescope:", format="%1f", value=16.)
        min_lon1 = st.sidebar.number_input("Longitude minute of the first telescope:", format="%1f", value=13.)
        sec_lon1 = st.sidebar.number_input("Longitude second of the first telescope:", format="%.1f", value=29.7)
        WE_lon1 = st.sidebar.selectbox("West or East:", ("W", "E"), index=1)

        lat_deg1 = str(degree_lat1) + " " + str(min_lat1) + " " + str(sec_lat1) + NS_lat1
        lon_deg1 = str(degree_lon1) + " " + str(min_lon1) + " " + str(sec_lon1) + WE_lon1

        lat_dec1 = dms_to_decimal(lat_deg1)
        lon_dec1 = dms_to_decimal(lon_deg1)

        if two_telescopes_location == "coordinates":

            degree_lat2 = st.sidebar.number_input("Latitude degree of second telescope:", format="%1f", value=23.)
            min_lat2 = st.sidebar.number_input("Latitude minute of second telescope:", format="%1f", value=20.)
            sec_lat2 = st.sidebar.number_input("Latitude second of second telescope:", format="%.1f", value=29.7)
            NS_lat2 = st.sidebar.selectbox("North or South: ", ("N", "S"), index=1)

            degree_lon2 = st.sidebar.number_input("Longitude degree of second telescope:", format="%1f", value=16.)
            min_lon2 = st.sidebar.number_input("Longitude minute of second telescope:", format="%1f", value=13.)
            sec_lon2 = st.sidebar.number_input("Longitude second of second telescope:", format="%.1f", value=28.1)
            WE_lon2 = st.sidebar.selectbox("West or East ", ("W", "E"), index=1)

            lat_deg2 = str(degree_lat2) + " " + str(min_lat2) + " " + str(sec_lat2) + NS_lat2
            lon_deg2 = str(degree_lon2) + " " + str(min_lon2) + " " + str(sec_lon2) + WE_lon2

            lat_dec2 = dms_to_decimal(lat_deg2)
            lon_dec2 = dms_to_decimal(lon_deg2)

if two_telescopes=="Yes":
    if two_telescopes_location == "baseline":
        baseline = st.sidebar.number_input("Enter the baseline used in meters:", format="%1f", help="The baseline vector should not include the difference in height, this is really the difference in N-S and W-E.")

if coordinates_form=="decimal degrees (DD)":
    if two_telescopes == "No":

        lat_dec1 = st.sidebar.number_input("Latitude degree:", format="%.5f", value=-23.3422040)
        lon_dec1 = st.sidebar.number_input("Latitude minute:", format="%.5f", value=16.2249443)

    if two_telescopes == "Yes":
        
        lat_dec1 = st.sidebar.number_input("Latitude degree for the first telescope:", format="%.5f", value=-23.3422040)
        lon_dec1 = st.sidebar.number_input("Latitude minute for the first telescope:", format="%.5f", value=16.2249443)
        
        if two_telescopes_location=="coordinates":
        
            lat_dec2 = st.sidebar.number_input("Latitude degree for the second telescope:", format="%.5f", value=-23.3415711)
            lon_dec2 = st.sidebar.number_input("Latitude minute for the second telescope:", format="%.5f", value=16.2244744)



utc_offset=st.sidebar.number_input("Enter UTC offset for correct estimate of visibility during the night:", min_value=-12, max_value=+12, step=1, value=+2)


# Define your conditions for data
if two_telescopes == "Yes" and two_telescopes_location == "coordinates":
    data = {'lat': [lat_dec1, lat_dec2], 'lon': [lon_dec1, lon_dec2]}
elif two_telescopes == "No" or two_telescopes_location == "baseline":
    data = {'lat': [lat_dec1], 'lon': [lon_dec1]}

map = st.sidebar.checkbox("Show map of locations of telescopes", help="This has no effect on any of the calculation and only serves to crosscheck your input of coordinates.")


if map:
    st.write("Here you see a map with the locations you gave coordinates to.")
    # Create DataFrame from data
    df = pd.DataFrame(data)

    # If two telescopes and location is 'baseline', generate points around lat_dec1, lon_dec1
    if two_telescopes == "Yes" and two_telescopes_location == "baseline":
        num_points = 100  # Number of points to generate around the circle
        baseline_meters = baseline  # Your baseline radius in meters

        # Convert baseline radius from meters to degrees
        baseline_radius_lat = baseline_meters / 111320  # Latitude degrees
        baseline_radius_lon = baseline_meters / (111320 * np.cos(np.radians(lat_dec1)))  # Longitude degrees

        angles = np.linspace(0, 2 * np.pi, num_points)
        lat_circle = lat_dec1 + baseline_radius_lat * np.cos(angles)
        lon_circle = lon_dec1 + baseline_radius_lon * np.sin(angles)

        # Add these points to the data dictionary
        df_circle = pd.DataFrame({'lat': lat_circle, 'lon': lon_circle})
        df = pd.concat([df, df_circle], ignore_index=True)

    # Display the map with the points
    st.map(df, size=1, zoom=15)

number_of_stars = st.sidebar.number_input("Number of brightest stars to check:", min_value=10, max_value=1600, help="Going through all of the 1500 stars can take up to 5 minutes, so here you can specify how" 
                                          "many of the brightest stars you want to check for a particular night. A recommended choice would be 50-100, this takes up to 30s to calculate.")

if two_telescopes=="Yes":
    heights = st.sidebar.checkbox("Enter heights above sea level", help="This is mainly relevant if the difference in height between telescopes is of similar or larger order than the baseline. In that case the UV-trace might turn out to be different.")

    if heights:
        height1=st.sidebar.number_input("Height of first telescope above sea level in meters:", format="%1f")
        height2=st.sidebar.number_input("Height of second telescope above sea level in meters:", format="%1f")

    else:
        height1 = 0
        height2 = 0

if two_telescopes=="Yes":
    if two_telescopes_location=="coordinates":
        # Calculate differences
        delta_lat = lat_dec2 - lat_dec1
        delta_lon = lon_dec2 - lon_dec1
        # Difference in height (x_up)
        delta_height = height2 - height1

        # Earth radius (assuming a spherical Earth)
        R = 6371.0 * u.km

        # Calculate differences in east (x_E) and north (x_N) directions
        x_E = np.round(((delta_lon * np.pi / 180.0) * R * np.cos(lat_dec1 * np.pi / 180.0)).value*1000, 3)
        x_N = np.round((delta_lat * R * np.pi / 180.0).value*1000, 3)
        x_up = delta_height

    # Difference in height (x_up)
    delta_height = height2 - height1
    x_up = delta_height

# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------


# Initialize a dictionary to store data from each column
data = {}

# Open the CSV file
with open(csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Read the header row

    # Create an empty list for each column
    for column_name in header:
        data[column_name] = []

    # Iterate over each row in the CSV file
    for row in reader:
        # Iterate over each item in the row and append it to the respective column list
        for idx, item in enumerate(row):
            column_name = header[idx]  # Get the corresponding column name
            data[column_name].append(item)


st.write("In the following table you see a list of stars that the app will run through to check which of those are visible. Right now you "
         "have selected to filter the brightest "+str(number_of_stars)+" stars to go through.")

df = pd.read_csv(csv_file)
st.write(df_display.head(number_of_stars))
date_obj = datetime.datetime.combine(date, datetime.time(12, 0))
date_str =str(date_obj)


# Convert the date object to a Julian date
date_JD = date_obj.toordinal() + 1721425 + .33333 - (
        1 / 24) * utc_offset  # added 1/3 since observations will most likely start at 8pm + offset of timezone


# Create a Time object from the observation time in Julian date
observation_time_utc = Time(date_JD, format='jd')


# Extract stars which are visible on the night of observation
RA = []
for star in data["RA_decimal"]:
    RA.append(star)

Dec = []
for star in data["Dec_decimal"]:
    Dec.append(star)

equatorial_coords = []
for i in range(len(data["RA_decimal"])):
    coord = SkyCoord(RA[i], Dec[i], unit=(u.hourangle, u.deg), frame='icrs')
    equatorial_coords.append(coord)

# Initialize lists to store altitude and azimuth values for each star
altitudes_per_star = []
azimuths_per_star = []
plt.style.use('default')
with st.form("key1"):
    # ask for input
    st.write("Click here to search for stars visible in the night of the ", str(date), ".")
    button_check_visible = st.form_submit_button("Search")

    if button_check_visible:
        # Iterate over each equatorial coordinate
        for equatorial_coord in tqdm(equatorial_coords[:number_of_stars]):
            # Initialize lists to store altitude and azimuth values for the current star
            altitudes = []
            azimuths = []

            # Define the time span
            hours_before = 0
            hours_after = 12
            start_time = observation_time_utc - TimeDelta(hours_before * u.hour)
            end_time = observation_time_utc + TimeDelta(hours_after * u.hour)

            # Calculate altitude and azimuth for each time point
            times = start_time + (end_time - start_time) * np.linspace(0, 1, 97)[:, None]
            for time in times:
                altaz_coords = equatorial_coord.transform_to(
                    AltAz(obstime=time, location=EarthLocation(lat=lat_dec1, lon=lon_dec1, height=height1)))
                altitude = altaz_coords.alt
                azimuth = altaz_coords.az
                altitudes.append(altitude)
                azimuths.append(azimuth)

            # Append altitude and azimuth lists for the current star to the main lists
            altitudes_per_star.append(altitudes)
            azimuths_per_star.append(azimuths)

        # Define a threshold for altitude entries
        altitude_threshold = 10  # Minimum altitude value during one night

        # Iterate over each star's altitude data
        extracted_data = {}
        for key in data.keys():
            extracted_data[key] = []

        for star_idx, altitudes in tqdm(enumerate(altitudes_per_star)):
            
            # Count the number of altitude entries less than the threshold
            low_altitude_count = sum(altitude.value < altitude_threshold for altitude in altitudes)

            # Check if more than 1/4 of the entries have altitudes less than the threshold
            if low_altitude_count < len(altitudes) / 4:
                # Remove corresponding entries from the data dictionary for the star
                for key in data.keys():
                    extracted_data[key].append(data[key][star_idx])
        




        output_file = 'stars_visible_' + date_str + '.csv'
        # Write the extracted data to the CSV file
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header row using the keys of the extracted_data dictionary
            writer.writerow(extracted_data.keys())

            # Write the data rows
            # Determine the number of rows based on the length of one of the lists in extracted_data
            num_rows = len(next(iter(extracted_data.values())))

            for i in range(num_rows):
                # Get the data for each column and write it to the CSV file
                row_data = [extracted_data[key][i] for key in extracted_data.keys()]
                writer.writerow(row_data)
            
        df = pd.read_csv(output_file)
        st.write("The stars that are visible in the night of "+str(date)+ " at your location are listed below. You can download the list as a .csv file.")
        st.write(df)





# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------


df_all_stars['Identifier'] = df_all_stars.apply(
    lambda x: f"{x['BayerF']} ({x['Common']})" if pd.notna(x['Common']) else x['BayerF'], axis=1)

# Create a selectbox for selecting a star

if two_telescopes=="Yes":
    st.write("Alternatively if you already have a star in mind, click here to select a star out of a drop-down menu and draw the visibility map and its trace along the sky.")

    if two_telescopes_location=="baseline":
        if baseline==0:
            st.write("Please enter a baseline that is non-zero to see the trace on the visibility map.")

        st.write("Indicate the orientation angle of the baseline.")
        angle = st.number_input("Angle of orientation in degrees:", min_value=0, max_value=359, value=90, help="i.e. 0° corresponds to $x_N$=baseline, $x_E$=0; 90° corresponds to $x_N$=0, $x_E$=baseline and so on.")

        x_E = np.sin(2*np.pi*angle/360)*baseline
        x_N = np.cos(2*np.pi*angle/360)*baseline

    selected_star = st.selectbox('Select a star:', options=df_all_stars['Identifier'], help="The stars are sorted by brightness i.e. the apparent magnitude")
    star_details = df_all_stars[df_all_stars['Identifier'] == selected_star].iloc[0]


    star_of_interest = f"{star_details['BayerF']},{star_details['Common']},{star_details['Parallax']},{star_details['Distance']},{star_details['Umag']},{star_details['Vmag']},{star_details['Bmag']},{star_details['Temp']},{star_details['RA_decimal']},{star_details['Dec_decimal']},{star_details['RA']},{star_details['Dec']},{star_details['Diameter_U']},{star_details['Diameter_V']},{star_details['Diameter_B']},{star_details['Phi_U']},{star_details['Phi_V']},{star_details['Phi_B']}"


    values = star_of_interest.split(',')
    BayerF = values[0]
    given_ra_decimal = float(values[8])
    given_dec_decimal = float(values[9])
    diameter_V = float(values[13])
    Phi_V = float(values[16])
    diameter_in_rad = diameter_V / 1000 * np.pi / (3600 * 180)

    lat = lat_dec1
    lon = lon_dec1


    # Convert the date object to a Julian date
    date_JD = date_obj.toordinal() + 1721425 + .33333 - (1 / 24) * utc_offset

    # Create a Time object from the observation time in Julian date
    observation_time_utc = Time(date_JD, format='jd')

    equatorial_coords = SkyCoord(given_ra_decimal, given_dec_decimal, unit=(u.hourangle, u.deg), frame='icrs')

    # Define time range for trail calculation
    hours_before = -1 / 3600
    hours_after = 12.001

    start_time = observation_time_utc - TimeDelta(hours_before * u.hour)
    end_time = observation_time_utc + TimeDelta(hours_after * u.hour)
    times = start_time + (end_time - start_time) * np.linspace(0, 1, 97)[:, None]

    # Calculate coordinates at each time step
    altitudes = []
    azimuths = []

    for time in times:
        altaz_coords = equatorial_coords.transform_to(
            AltAz(obstime=time, location=EarthLocation(lat=lat, lon=lon, height=height1)))
        altitudes.append(altaz_coords.alt)
        azimuths.append(altaz_coords.az)

    # Convert lists to arrays
    altitudes = np.array(altitudes)
    azimuths = np.array(azimuths)
    azimuths_flat = azimuths.flatten()
    datetime_objects = [Time(time[0]).to_datetime() for time in times]

    # Extract only the time component from datetime objects and convert to string
    time_components = [dt.time().strftime('%H:%M') for dt in datetime_objects]

    # Create columns for the plots
    col1, col2 = st.columns(2)

    # Plot the trail in the first column
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sc = ax1.scatter(time_components, altitudes, c=azimuths_flat)
    plt.colorbar(sc, label='Azimuth [°]', ax=ax1)  # Add color bar indicating azimuth
    ax1.set_xticks(time_components[::12])
    ax1.set_title("Celestial Path of "+ BayerF)
    ax1.set_xlabel('UTC Time')
    ax1.set_ylabel('Altitude [°]')
    ax1.set_ylim(0, 90)
    ax1.grid(True)
    col1.pyplot(fig1)

    if selected_star=="ι Orionis (Nair Al Saif)" or selected_star=="α Virginis (Spica)":
        st.write("You have selected a binary star, which means that the visibility map is going to vary as a function of time.")
        st.write("To see how the visibility map changes, you have now a slider to select the frame.")

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        spica = Orbit(P=4.01, e=0.123, I=63,
            Omega=309.938, omega=2136.727, jdperi=2440678.008, q=0.6188)             
        
        if selected_star=="ι Orionis (Nair Al Saif)":
            R_in_sol1 = 8.3
            R_in_sol2 = 5.4
            distance_in_pc= 412
            temp = ['#9fbfff','#a1c0ff']
            R=[apparent_radius_mas(R_in_sol1, distance_in_pc), apparent_radius_mas(R_in_sol2, distance_in_pc)]

            Omega_ori = st.number_input("Enter longitude of ascending node in degrees:", value=0, min_value=0, max_value=359)
            omega_ori = st.number_input("Enter argument of periapsis:", value=0, min_value=0, max_value=359)

            IotaOrionis=Orbit(P=29.1338, e=0.764, I=60,
            Omega=Omega_ori, omega=omega_ori, jdperi=2450072.80, q=(23.1/13.1))
            binary_selected=IotaOrionis

        if selected_star=="α Virginis (Spica)":
            R_in_sol1 = 7.4
            R_in_sol2 = 3.74
            distance_in_pc = 77
            temp = ['#a2c1ff','#a7c4ff']
            R=[apparent_radius_mas(R_in_sol1, distance_in_pc), apparent_radius_mas(R_in_sol2, distance_in_pc)]
            spica = Orbit(P=4.01, e=0.123, I=63,
                Omega=309.938, omega=2136.727, jdperi=2440678.008, q=0.6188)  
            binary_selected=spica

        if two_telescopes_location=="baseline":
            if baseline==0:
                plt.text(0.8, 0.6, "Please enter a non-zero baseline", size=10, rotation=0.,
                ha="right", va="top",
                bbox=dict(boxstyle="square",
                    ec=(1., 0.5, 0.5),
                    fc=(1., 0.8, 0.8),
                    )
                )
                ax2.set_title("Visibility Map of "+ BayerF + ", diameter: " + str(diameter_V) + " mas\n "
                            "$\Phi$ = " + str(np.round(Phi_V, 7)) + " photons m$^{-2}$ s$^{-1}$ Hz$^{-1}$")
                ax2.set_xlabel('U [m]')
                ax2.set_ylabel('V [m]')
                ax2.set_aspect('equal')
                col2.pyplot(fig2)

        ds = 6e-10
        N = 2048
        sx,sy,x,y = grids(ds,N,lambda_V)

        scl = 8e-9 # orbital a in radians
        mscl = scl*(3000/np.pi*60**3)
        
        F = len(times)  # Total frames are now determined by the length of 'times'

        fr = st.slider(
            "Select the frame",
            0,
            F - 1,  # Adjust slider range to be 0-indexed
            value=10,
            step=1,
            help="This slider controls which moment in the night is displayed."
        )

        # Access the current_time directly from the 'times' array
        current_time = Time(times[fr][0]).to_datetime() 
        st.write(f"Selected time: {current_time.strftime('%H:%M')}")

        jd = times[fr][0].jd 
        xp, yp = binary_selected.binarypos(jd-0.1)
        dis = scl * ((xp[0] - xp[1]) ** 2 + (yp[0] - yp[1]) ** 2) ** (1/2)
        phi = np.arctan2(yp[1], xp[1])
        sig = visib(x, y, lambda_V, dis, phi)  # Assuming visib() is defined
        sig /= np.max(sig)

        # Subplot 2
        cs= draw(x, y, sig, 2, 'ground', ax2, cmap='gray') 

        mark_size=15
        edge=0.5

        U = []
        V = []
        W = []
        
        for time in times:
            HA_value = RA_2_HA(given_ra_decimal, time.jd)
            matrices = R_x(given_dec_decimal * u.deg).dot(R_y(HA_value * u.deg)).dot(R_x(-lat * u.deg))
            h_plane = np.array([[x_E], [x_N], [x_up]])
            UVW_plane = matrices.dot(h_plane)
            
            U.append(UVW_plane[0][0])
            V.append(UVW_plane[1][0])
            W.append(UVW_plane[2][0])
        
        ax2.plot(U, V, '.', color='gold', markeredgecolor='black', markersize=mark_size, markeredgewidth=edge)
        ax2.plot()

        U_neg = [-u for u in U]
        V_neg = [-v for v in V]


        ax2.plot(U_neg, V_neg, '.', color='gold', markeredgecolor='black', markersize=mark_size, markeredgewidth=edge)

        U = []
        V = []
        W = []

        # Create a grid of points
        resolution = 300
        size_to_plot = np.sqrt(x_E**2 + x_N**2 + x_up**2)
        x = np.linspace(-size_to_plot, size_to_plot, resolution)
        y = np.linspace(-size_to_plot, size_to_plot, resolution)
        col=temp
        X, Y = np.meshgrid(x, y)


        # Difference in height (x_up)
        delta_height = height2 - height1
        x_up = delta_height
        
        HA_value = RA_2_HA_single(given_ra_decimal, jd)
        matrices = R_x(given_dec_decimal * u.deg).dot(R_y(HA_value * u.deg)).dot(R_x(-lat * u.deg))
        h_plane = np.array([[x_E], [x_N], [x_up]])
        UVW_plane = matrices.dot(h_plane)
            
        U = UVW_plane[0, 0]
        V = UVW_plane[1, 0]
        W = UVW_plane[2, 0]

        # Plot the UVW plane in the second column

        ax2.plot(U, V, '.', color='red', markeredgecolor='black', markersize=mark_size, markeredgewidth=edge)
        ax2.plot(-U, -V, '.', color='red', markeredgecolor='black', markersize=mark_size, markeredgewidth=edge)

        ax2.set_title("Visibility Map of "+ BayerF + ", diameter: " + str(diameter_V) + " mas\n "
                        "$\Phi$ = " + str(np.round(Phi_V, 7)) + " photons m$^{-2}$ s$^{-1}$ Hz$^{-1}$")
        ax2.set_xlabel('U [m]')
        ax2.set_ylabel('V [m]')
        ax2.set_aspect('equal')
        plt.colorbar(cs, ax=ax2, label="Intensity")



        # Plot the UVW plane in the second column
        if two_telescopes_location=="baseline":
            if baseline==0:
                plt.text(0.8, 0.6, "Please enter a non-zero baseline", size=10, rotation=0.,
                ha="right", va="top",
                bbox=dict(boxstyle="square",
                    ec=(1., 0.5, 0.5),
                    fc=(1., 0.8, 0.8),
                    )
                )
                ax2.set_title("Visibility Map of "+ BayerF + ", diameter: " + str(diameter_V) + " mas\n "
                            "$\Phi$ = " + str(np.round(Phi_V, 7)) + " photons m$^{-2}$ s$^{-1}$ Hz$^{-1}$")
                ax2.set_xlabel('U [m]')
                ax2.set_ylabel('V [m]')
                ax2.set_aspect('equal')
                col2.pyplot(fig2)

        
        
        if two_telescopes_location=="baseline":
            if baseline>0:
                col2.pyplot(fig2)
        if two_telescopes_location=="coordinates":
            col2.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(8, 6))
        
        # Subplot 3
        ax3.set_xlim((-mscl, mscl))
        ax3.set_ylim((-mscl, mscl))
        ax3.set_aspect('equal')

        for k in range(2):
            ax3.add_patch(plt.Circle((mscl * xp[k], mscl * yp[k]), mscl * R[k], color=col[k]))
        ax3.set_xlabel('mas')
        ax3.set_ylabel('mas')
        ax3.set_title("Positions of the binaries "+ BayerF)
        st.pyplot(fig3)
        
    else:


        U = []
        V = []
        W = []
        
        # Create a grid of points
        resolution = 300
        size_to_plot = np.sqrt(x_E**2 + x_N**2 + x_up**2)
        x = np.linspace(-size_to_plot, size_to_plot, resolution)
        y = np.linspace(-size_to_plot, size_to_plot, resolution)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X ** 2 + Y ** 2)


        # Difference in height (x_up)
        delta_height = height2 - height1
        x_up = delta_height
        
        for time in times:
            HA_value = RA_2_HA(given_ra_decimal, time.jd)
            matrices = R_x(given_dec_decimal * u.deg).dot(R_y(HA_value * u.deg)).dot(R_x(-lat * u.deg))
            h_plane = np.array([[x_E], [x_N], [x_up]])
            UVW_plane = matrices.dot(h_plane)
            
            U.append(UVW_plane[0][0])
            V.append(UVW_plane[1][0])
            W.append(UVW_plane[2][0])
        
        intensity_values = visibility(R, diameter_in_rad, (5.4e-7))  # wavelength in meters
        
        # Plot the UVW plane in the second column
        fig2, ax2 = plt.subplots()
        if two_telescopes_location=="baseline":
            if baseline==0:
                plt.text(0.8, 0.6, "Please enter a non-zero baseline", size=10, rotation=0.,
                ha="right", va="top",
                bbox=dict(boxstyle="square",
                    ec=(1., 0.5, 0.5),
                    fc=(1., 0.8, 0.8),
                    )
                )
                ax2.set_title("Visibility Map of "+ BayerF + ", diameter: " + str(diameter_V) + " mas\n "
                            "$\Phi$ = " + str(np.round(Phi_V, 7)) + " photons m$^{-2}$ s$^{-1}$ Hz$^{-1}$")
                ax2.set_xlabel('U [m]')
                ax2.set_ylabel('V [m]')
                ax2.set_aspect('equal')
                col2.pyplot(fig2)

        cax = ax2.imshow(intensity_values, norm=None, extent=(-size_to_plot, size_to_plot, -size_to_plot, size_to_plot), origin='lower', cmap='gray')
        ax2.plot(U, V, '.', color='gold', markeredgecolor='black')

        U_neg = [-u for u in U]
        V_neg = [-v for v in V]


        ax2.plot(U_neg, V_neg, '.', color='gold', markeredgecolor='black')

        ax2.set_title("Visibility Map of "+ BayerF + ", diameter: " + str(diameter_V) + " mas\n "
                        "$\Phi$ = " + str(np.round(Phi_V, 7)) + " photons m$^{-2}$ s$^{-1}$ Hz$^{-1}$")
        ax2.set_xlabel('U [m]')
        ax2.set_ylabel('V [m]')
        ax2.set_aspect('equal')
        plt.colorbar(cax, label="Intensity")
        if two_telescopes_location=="baseline":
            if baseline>0:
                col2.pyplot(fig2)
        if two_telescopes_location=="coordinates":
            col2.pyplot(fig2)

