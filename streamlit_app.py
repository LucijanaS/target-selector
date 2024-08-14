import json
import csv
from brightstar_functions import *
import streamlit as st
import pandas as pd

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
brightest_stars = sorted(stars_data, key=lambda x: x['Vmag'])[:n_brightest_stars]

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



# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------



c1, c2 = st.columns(2)
with c1:
    month = st.selectbox(
    "Choose month of observation",
    ("January", "February", "March", "April", "May", "June", 
     "July", "August", "September", "October", "November", "December")
)
with c2:
    year = st.number_input("Enter Year of observation:", value=2024, step=1)


month_number = {
    "January": "01",
    "February": "02",
    "March": "03",
    "April": "04",
    "May": "05",
    "June": "06",
    "July": "07",
    "August": "08",
    "September": "09",
    "October": "10",
    "November": "11",
    "December": "12"
}[month]

c1, c2, c3 = st.columns(3)
with c1:
    coordinates_form = st.radio(
        "Enter coordinates in: ",
        ["degrees, minute, second (DMS)", "decimal degrees (DD)"])
with c2:
    two_telescopes = st.radio(
        "Two telescopes? ",
        ["Yes", "No"])
    if two_telescopes=="Yes":
        with c3:
            two_telescopes_location = st.radio(
                "Enter  second telescope as baseline or coordinate",
                ["baseline", "coordinates"])


if coordinates_form == "degrees, minute, second (DMS)":
    if two_telescopes=="No":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            degree_lat1 = st.number_input("Latitude degree", format="%1f")
        with c2:
            min_lat1 = st.number_input("Latitude minute", format="%1f")
        with c3:
            sec_lat1 = st.number_input("Latitude second", format="%.1f")
        with c4:
            NS_lat1 = st.selectbox("North or South", ("N", "S"))
    
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            degree_lon1 = st.number_input("Longitude degree", format="%1f")
        with c2:
            min_lon1 = st.number_input("Longitude minute", format="%1f")
        with c3:
            sec_lon1 = st.number_input("Longitude second", format="%.1f")
        with c4:
            WE_lon1 = st.selectbox("West or East", ("W", "E"))
        height1=st.number_input("Height above sea level in meters:", value=0)
    
        lat_deg1=str(degree_lat1)+" "+str(min_lat1)+" "+str(sec_lat1)+NS_lat1
        lon_deg1=str(degree_lon1)+" "+str(min_lon1)+" "+str(sec_lon1)+WE_lon1

        lat_dec1 = dms_to_decimal(lat_deg1)
        lon_dec1 = dms_to_decimal(lon_deg1)

    if two_telescopes == "Yes":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            degree_lat1 = st.number_input("Latitude degree of the first telescope", format="%1f")
        with c2:
            min_lat1 = st.number_input("Latitude minute of the first telescope", format="%1f")
        with c3:
            sec_lat1 = st.number_input("Latitude second of the first telescope", format="%.1f")
        with c4:
            NS_lat1 = st.selectbox("North or South", ("N", "S"))
    
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            degree_lon1 = st.number_input("Longitude degree of the first telescope", format="%1f")
        with c2:
            min_lon1 = st.number_input("Longitude minute of the first telescope", format="%1f")
        with c3:
            sec_lon1 = st.number_input("Longitude second of the first telescope", format="%.1f")
        with c4:
            WE_lon1 = st.selectbox("West or East", ("W", "E"))
        height1=st.number_input("Height of first telescope above sea level in meters:", value=0)
    
        lat_deg1=str(degree_lat1)+" "+str(min_lat1)+" "+str(sec_lat1)+NS_lat1
        lon_deg1=str(degree_lon1)+" "+str(min_lon1)+" "+str(sec_lon1)+WE_lon1

        lat_dec1 = dms_to_decimal(lat_deg1)
        lon_dec1 = dms_to_decimal(lon_deg1)

        
        
        if two_telescopes_location=="coordinates":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                degree_lat2 = st.number_input("Latitude degree of second telescope", format="%1f")
            with c2:
                min_lat2 = st.number_input("Latitude minute of second telescope", format="%1f")
            with c3:
                sec_lat2 = st.number_input("Latitude second of second telescope", format="%.1f")
            with c4:
                NS_lat2 = st.selectbox("North or South ", ("N", "S"))
    
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                degree_lon2 = st.number_input("Longitude degree of second telescope", format="%1f")
            with c2:
                min_lon2 = st.number_input("Longitude minute of second telescope", format="%1f")
            with c3:
                sec_lon2 = st.number_input("Longitude second of second telescope", format="%.1f")
            with c4:
                WE_lon2 = st.selectbox("West or East ", ("W", "E"))
    
                lat_deg2=str(degree_lat2)+" "+str(min_lat2)+" "+str(sec_lat2)+NS_lat2
                lon_deg2=str(degree_lon2)+" "+str(min_lon2)+" "+str(sec_lon2)+WE_lon2

                lat_dec2 = dms_to_decimal(lat_deg2)
                lon_dec2 = dms_to_decimal(lon_deg2)
        else:
            baseline=st.number_input("Enter the baseline used in meters", format="%1f")
        height2=st.number_input("Height of second telescope above sea level in meters:", value=0)





if coordinates_form=="decimal degrees (DD)":
    if two_telescopes == "No":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            lat_dec1 = st.number_input("Latitude degree", format="%.5f")
        with c2:
            lon_dec1 = st.number_input("Latitude minute", format="%.5f")
        height1=st.number_input("Height above sea level in meters: ", value=0)

    if two_telescopes == "Yes":
        c1, c2 = st.columns(2)
        with c1:
            lat_dec1 = st.number_input("Latitude degree for the first telescope", format="%.5f")
        with c2:
            lon_dec1 = st.number_input("Latitude minute for the first telescope", format="%.5f")
        
        if two_telescopes_location=="coordinates":
        
            c1, c2 = st.columns(2)
            with c1:
                lat_dec2 = st.number_input("Latitude degree for the second telescope", format="%.5f")
            with c2:
                lon_dec2 = st.number_input("Latitude minute for the second telescope", format="%.5f")
        else:
            baseline=st.number_input("Enter the baseline used in meters", format="%1f")
        height2=st.number_input("Height above sea level in meters: ", value=0)

utc_offset=st.number_input("Enter UTC offset for correct estimate of visibility during the night", min_value=-12, max_value=+12, step=1, value=0)


if two_telescopes=="Yes" and two_telescopes_location=="coordinates":
    data = {'lat': [lat_dec1, lat_dec2], 'lon': [lon_dec1, lon_dec2]}

if two_telescopes=="No" or two_telescopes_location=="baseline":
    data = {'lat': [lat_dec1], 'lon': [lon_dec1]}

map = st.checkbox("Show map of locations of telescopes")

if map:
    df = pd.DataFrame(data)
    st.map(df, size=1)


number_of_stars = st.number_input("Number of brightest stars to check:", min_value=10, max_value=1600)

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
        x_E = np.round(((delta_lon * np.pi / 180.0) * R * np.cos(lat1 * np.pi / 180.0)).value*1000, 3)
        x_N = np.round((delta_lat * R * np.pi / 180.0).value*1000, 3)
        x_up = delta_height.value

    if two_telescopes_location=="baseline":
        x_E=baseline
        x_N=0

        # Difference in height (x_up)
        delta_height = height2 - height1
        x_up = delta_height

# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------


# Initialize a dictionary to store data from each column
data = {}

# Open the CSV file
with open('10000stars_data.csv', newline='') as csvfile:
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

date_str=str(year)+"-"+str(month_number)+"-"+"15"

# Parse the input string into a datetime object
date_obj = datetime.strptime(date_str, '%Y-%m-%d')

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

with st.form("key1"):
    # ask for input
    st.write("Click here to search for stars visible at night in the month of ", month, ".")
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

    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for star_idx, altitudes in tqdm(enumerate(altitudes_per_star)):
        
        my_bar.progress(star_idx + 1, text=progress_text)
        # Count the number of altitude entries less than the threshold
        low_altitude_count = sum(altitude.value < altitude_threshold for altitude in altitudes)

        # Check if more than 1/4 of the entries have altitudes less than the threshold
        if low_altitude_count < len(altitudes) / 4:
            # Remove corresponding entries from the data dictionary for the star
            for key in data.keys():
                extracted_data[key].append(data[key][star_idx])
    my_bar.empty()



    output_file = 'stars_visible_' + date_str + '.csv'
    df = pd.read_csv(output_file)
    st.write(df)
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





# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------


df_all_stars['Identifier'] = df_all_stars.apply(
    lambda x: f"{x['BayerF']} ({x['Common']})" if pd.notna(x['Common']) else x['BayerF'], axis=1)

# Create a selectbox for selecting a star

with st.form("key2"):
    st.write("Alternatively click here to select a star out of a drop-down menu and draw the visibility map and its trace along the sky.")
    button_check_select = st.form_submit_button("Select")
    
    if button_check_select:
        selected_star = st.selectbox('Select a star:', placeholder="Select contact method...", options=df_all_stars['Identifier'])
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
        
        # Parse the input string into a datetime object
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Convert the date object to a Julian date
        date_JD = date_obj.toordinal() + 1721425 + .33333 - (1 / 24) * utc_offset
        
        # Create a Time object from the observation time in Julian date
        observation_time_utc = Time(date_JD, format='jd')
        
        equatorial_coords = SkyCoord(given_ra_decimal, given_dec_decimal, unit=(u.hourangle, u.deg), frame='icrs')
        
        # Define time range for trail calculation
        hours_before = -1 / 3600
        hours_after = 12.001 if two_telescopes_location == "coordinates" else 24.001
        
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
        fig1, ax1 = plt.subplots()
        sc = ax1.scatter(time_components, altitudes, c=azimuths_flat)
        plt.colorbar(sc, label='Azimuth [°]', ax=ax1)  # Add color bar indicating azimuth
        ax1.set_xticks(time_components[::12])
        ax1.set_title(BayerF)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Altitude [°]')
        ax1.set_ylim(0, 90)
        ax1.grid(True)
        col1.header("Altitude vs Time")
        col1.pyplot(fig1)
        
        U = []
        V = []
        W = []
        
        # Create a grid of points
        resolution = 300
        size_to_plot = np.sqrt(x_E**2 + x_N**2)
        x = np.linspace(-size_to_plot, size_to_plot, resolution)
        y = np.linspace(-size_to_plot, size_to_plot, resolution)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X ** 2 + Y ** 2)
        
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
        cax = ax2.imshow(intensity_values, norm=None, extent=(-size_to_plot, size_to_plot, -size_to_plot, size_to_plot), origin='lower', cmap='gray')
        ax2.plot(U, V, '.', color='gold', markeredgecolor='black')
        ax2.set_title(BayerF + " diameter: " + str(diameter_V) + " mas\n "
                          "$\Phi$ = " + str(np.round(Phi_V, 7)) + " photons m$^{-2}$ s$^{-1}$ Hz$^{-1}$")
        ax2.set_xlabel('U [m]')
        ax2.set_ylabel('V [m]')
        ax2.set_aspect('equal')
        plt.colorbar(cax, label="Intensity")
        col2.pyplot(fig2)
