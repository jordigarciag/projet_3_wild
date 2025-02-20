import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from geopy.geocoders import Nominatim
except ImportError:
    
    install_package('geopy')
    from geopy.geocoders import Nominatim

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


# Configuration de la page
st.set_page_config(
    page_title="My Cathedra",
    page_icon="🎭",
    layout="wide")

# En-tête simple centré
st.markdown("<h1 style='text-align: center'>🎼 My Cathedra 🎻</h1>", unsafe_allow_html=True)

def get_coordinates(address):
    try:
        geolocator = Nominatim(user_agent="my_cathedra")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        location = geocode(address)
        if location:
            return pd.Series([location.latitude, location.longitude])
        return pd.Series([None, None])
    except:
        return pd.Series([None, None])

# Menu de navigation épuré
selected = option_menu(
    menu_title=None,
    options=["Analyses", "Assistant", "Suggestions"],
    icons=["graph-up", "person", "lightbulb"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0.5rem", "background-color": "#f8f9fa", "border-radius": "4px"},
        "icon": {"color": "#2c3e50"},
        "nav-link": {"color": "#2c3e50", "font-size": "0.9rem", "text-align": "center"},
        "nav-link-selected": {"background-color": "#3498db", "color": "white"}
    })

def create_interactive_chart(data, title, is_participant=False):
    if is_participant:
        fig = go.Figure(data=[go.Bar(
            y=data['nom_complet'].head(10),
            x=data['nombre_evenements'].head(10),
            orientation='h',
            text=data['nombre_evenements'].head(10),
            textposition='auto',
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title="Nombre d'événements",
            yaxis_title="Participants",
            width=800,
            height=500,
            showlegend=False,
            yaxis={'autorange': 'reversed'}
        )
    else:
        fig = go.Figure(data=[go.Bar(
            x=data.index,
            y=data.values,
            text=data.values,
            textposition='auto',
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title="Année",
            yaxis_title="Nombre d'événements",
            width=800,
            height=500,
            showlegend=False
        )
    return fig

def create_pie_chart(data, title):
    fig = go.Figure(data=[go.Pie(
        labels=data.index,
        values=data.values,
        hole=0.3,
        textinfo='percent+label'
    )])
    
    fig.update_layout(
        title=title,
        width=800,
        height=600
    )
    return fig

def load_data():
    url = "https://raw.githubusercontent.com/jordigarciag/projet_3_wild/main/data.zip"
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    
    with zip_file.open('data.csv') as file:
        df = pd.read_csv(file, low_memory=False)
    return df

if selected == "Analyses":
    col1, col2 = st.columns([1, 3])
    with col1:
        options = {
            "events": "📊 Événements",
            "genres": "🎵 Genres",
            "occupancy": "🏆 Classement",
            "tickets": "🎫 Billetterie",
            "venues": "🏛️ Lieux",
            "participants": "👥 Participants"
        }
        for key, value in options.items():
            if st.button(value, use_container_width=True):
                keys_to_remove = [k for k in st.session_state.keys() if k.startswith('selected_')]
                for k in keys_to_remove:
                    del st.session_state[k]
                st.session_state.current_analysis = key

    with col2:
        if 'current_analysis' not in st.session_state:
            st.markdown("<h2 style='text-align: center; color: #756AB0;'>👋 Bienvenue !</h2>", unsafe_allow_html=True)
            st.write("<div style='text-align: center'>Sélectionnez une catégorie dans le menu de gauche pour démarrer l'exploration de vos données Weezevent.</div>", unsafe_allow_html=True)
        elif st.session_state.current_analysis:
            st.markdown("### " + options[st.session_state.current_analysis])
            if st.session_state.current_analysis in ["events", "genres"]:
                try:
                    df = load_data()
                    df['year'] = pd.to_datetime(df['event_start_date']).dt.year
                    years = sorted(df['year'].unique(), reverse=True)
                    years.insert(0, "Toutes les années")
                    all_genres = []
                    for genres in df['Genre'].dropna():
                        all_genres.extend([g.strip() for g in genres.split(',')])
                    unique_genres = sorted(list(set(all_genres)))
                    unique_genres.insert(0, "Tous les genres")

                    if st.session_state.current_analysis == "events":
                        selected_genre = st.selectbox('Genre', unique_genres, key='selected_genre_events')
                        filtered_df = df.copy()
                        if selected_genre != "Tous les genres":
                            filtered_df = filtered_df[filtered_df['Genre'].str.contains(selected_genre, na=False)]
                        events_count = filtered_df.groupby('year')['event_title'].nunique()
                        fig = create_interactive_chart(
                            events_count,
                            f'Nombre d\'événements par année{" - " + selected_genre if selected_genre != "Tous les genres" else ""}'
                        )
                        st.plotly_chart(fig)

                        st.info("Sélectionnez une année pour afficher sa programmation dans l'ordre chronologique 🗓️")
                        selected_year = st.selectbox('Année', years, key='selected_year_events')
                        if selected_year != "Toutes les années":
                            year_filtered_df = filtered_df[filtered_df['year'] == selected_year]
                            st.markdown(f"#### Événements de {selected_year}")
                            events_df = year_filtered_df[['event_title', 'event_start_date', 'event_description_clean']].drop_duplicates()
                            # Ajout du tri par date
                            events_df['event_start_date'] = pd.to_datetime(events_df['event_start_date'])
                            events_df = events_df.sort_values('event_start_date', ascending=True)
                            for i, (_, row) in enumerate(events_df.iterrows(), 1):
                                event_date = pd.to_datetime(row['event_start_date']).strftime('%d/%m/%Y')
                                st.write(f"{i}. {row['event_title']} ({event_date})")
                                if pd.notna(row['event_description_clean']):
                                    with st.expander("Voir la description de l'événement"):
                                        st.markdown(row['event_description_clean'])

                    elif st.session_state.current_analysis == "genres":
                        selected_year = st.selectbox('Année', years, key='selected_year_genres')
                        filtered_df = df.copy()
                        if selected_year != "Toutes les années":
                            filtered_df = filtered_df[filtered_df['year'] == selected_year]
                        df_genres = filtered_df.assign(Genre=filtered_df['Genre'].str.split(',')).explode('Genre')
                        df_genres['Genre'] = df_genres['Genre'].str.strip()
                        genre_counts = df_genres.groupby('Genre')['event_id'].nunique().sort_values(ascending=False)
                        fig = create_pie_chart(
                            genre_counts,
                            f'Distribution des genres{" en " + str(selected_year) if selected_year != "Toutes les années" else ""}'
                        )
                        st.plotly_chart(fig)

                        st.info("Veuillez indiquer un genre pour voir les événements associés 🎭")
                        selected_genre = st.selectbox('Genre', unique_genres, key='selected_genre_filter')
                        if selected_genre != "Tous les genres":
                            filtered_by_genre = filtered_df[filtered_df['Genre'].str.contains(selected_genre, na=False)]
                            if not filtered_by_genre.empty:
                                st.markdown(f"#### Événements du genre {selected_genre}")
                                events_df = filtered_by_genre[['event_title', 'event_start_date', 'event_description_clean']].drop_duplicates()
                                events_df['event_start_date'] = pd.to_datetime(events_df['event_start_date'])
                                events_df = events_df.sort_values('event_start_date', ascending=True)
                                for i, (_, row) in enumerate(events_df.iterrows(), 1):
                                    event_date = row['event_start_date'].strftime('%d/%m/%Y')
                                    st.write(f"{i}. {row['event_title']} ({event_date})")
                                    if pd.notna(row['event_description_clean']):
                                        with st.expander("Voir la description de l'événement"):
                                            st.markdown(row['event_description_clean'])

                except Exception as e:
                    st.error(f"Erreur lors de l'analyse: {e}")

            elif st.session_state.current_analysis == "participants":
                try:
                    df = load_data()
                    df['year'] = pd.to_datetime(df['event_start_date']).dt.year
                    years = sorted(df['year'].unique())
                    years.insert(0, "Toutes les années")
                    df['nom_complet'] = df['first_name'] + ' ' + df['last_name']
                    df = df.dropna(subset=['first_name', 'last_name'])
                    df = df[~df['nom_complet'].str.contains('Na|endeur', case=False, na=False)]
                    
                    st.warning("Dans le cadre de ce projet et conformément au Règlement Général sur la Protection des Données (RGPD), les données personnelles des participants ont été anonymisées. Toute information permettant d'identifier directement ou indirectement les participants a été supprimée ou modifiée pour garantir leur confidentialité.")
                    
                    selected_year = st.selectbox('Année', years, key='selected_year_participants')
                    
                    if selected_year != "Toutes les années":
                        year_data = df[df['year'] == selected_year]
                        participants_count = year_data.groupby('nom_complet')['event_id'].nunique().reset_index(name='nombre_evenements')
                        participants_count = participants_count.sort_values('nombre_evenements', ascending=False)
                        fig = create_interactive_chart(
                            participants_count,
                            f"Top 10 participants en {selected_year}",
                            is_participant=True
                        )
                        st.plotly_chart(fig)
                    else:
                        total_events = df.groupby('nom_complet')['event_id'].nunique().reset_index(name='nombre_evenements')
                        total_events = total_events.sort_values('nombre_evenements', ascending=False)
                        fig = create_interactive_chart(
                            total_events,
                            "Top 10 participants toutes années confondues",
                            is_participant=True
                        )
                        st.plotly_chart(fig)
                        
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse des participants: {e}")

            elif st.session_state.current_analysis == "occupancy":
                try:
                    df = load_data()
                    df['year'] = pd.to_datetime(df['event_start_date']).dt.year
                    years = sorted(df['year'].unique(), reverse=True)
                    years.insert(0, "Toutes les années")
                    
                    # Filtres en haut
                    selected_year = st.selectbox('Année', years, key='selected_year_occupancy')
                    
                    # Filtrer les données selon l'année
                    filtered_df = df.copy()
                    if selected_year != "Toutes les années":
                        filtered_df = filtered_df[filtered_df['year'] == selected_year]
                    
                    filtered_df['Genre'] = filtered_df['Genre'].str.split(',').str[0]
                    
                    # Créer le Top 10 des événements
                    performance_df = filtered_df[['event_title', 'billets_valides', 'Jauge finale', 'Taux de remplissage (%)', 'event_start_date', 'Genre']]
                    performance_grouped = performance_df.groupby('event_title').agg({
                        'billets_valides': 'mean',
                        'Jauge finale': 'mean',
                        'Taux de remplissage (%)': 'mean',
                        'event_start_date': 'first',
                        'Genre': 'first'
                    }).reset_index()
                    
                    top_10 = performance_grouped.sort_values('Taux de remplissage (%)', ascending=False).head(10)
                    
                    # Premier graphique - sans coloration par genre
                    fig_top10 = px.bar(top_10,
                                    x='event_title',
                                    y='Taux de remplissage (%)',
                                    title='Classement des événements par taux de remplissage',
                                    custom_data=['billets_valides', 'Jauge finale', 'event_start_date', 'Genre'])
                    
                    fig_top10.update_traces(
                        marker_color='#4B4BFF',  # Couleur unique pour toutes les barres
                        opacity=0.9,
                        hovertemplate="""
                        <b>%{x}</b><br>
                        Date: %{customdata[2]}<br>
                        Genre: %{customdata[3]}<br>
                        Taux de remplissage: %{y:.1f}%<br>
                        Billets validés: %{customdata[0]:,.0f}<br>
                        Jauge: %{customdata[1]:,.0f}
                        """
                    )
                    
                    fig_top10.update_layout(
                        xaxis_tickangle=-45,
                        xaxis_title="Événement",
                        yaxis_title="Taux de remplissage moyen (%)",
                        showlegend=False,
                        hoverlabel=dict(
                            font_size=14,
                            align='left'
                        ),
                        plot_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig_top10)
                    
                    # Analyse par genre
                    genre_performance_df = filtered_df[['Genre', 'billets_valides', 'Jauge finale', 'Taux de remplissage (%)']]
                    genre_grouped = genre_performance_df.groupby('Genre').agg({
                        'billets_valides': 'mean',
                        'Jauge finale': 'mean',
                        'Taux de remplissage (%)': 'mean'
                    }).reset_index()
                    
                    top_10_genres = genre_grouped.sort_values('Taux de remplissage (%)', ascending=False).head(10)
                    
                    # Créer un mapping des genres vers des couleurs plus vives
                    unique_genres = filtered_df['Genre'].unique()
                    colors = ['#FF4B4B', '#4B4BFF', '#FFB74B', '#4BFF4B', '#FF4BFF', 
                            '#4BFFFF', '#FFE74B', '#A64BFF', '#FF4B87', '#4BFF87']
                    color_map = dict(zip(unique_genres, colors[:len(unique_genres)]))
                    
                    # Deuxième graphique avec coloration par genre
                    fig_genres = px.bar(top_10_genres,
                                    x='Genre',
                                    y='Taux de remplissage (%)',
                                    color='Genre',
                                    color_discrete_map=color_map,
                                    title=f'Classement des genres par taux de remplissage{" en " + str(selected_year) if selected_year != "Toutes les années" else ""}',
                                    custom_data=['Jauge finale'])
                    
                    fig_genres.update_traces(
                        opacity=0.9,
                        hovertemplate="""
                        <b>%{x}</b><br>
                        Taux de remplissage: %{y:.1f}%<br>
                        Jauge moyenne: %{customdata[0]:,.0f}
                        """
                    )
                    
                    fig_genres.update_layout(
                        xaxis_tickangle=-45,
                        xaxis_title="Genre",
                        yaxis_title="Taux de remplissage (%)",
                        showlegend=False,
                        hoverlabel=dict(
                            font_size=14,
                            align='left'
                        ),
                        plot_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig_genres)
                    
                    # Graphique à bulles avec coloration par genre
                    fig = px.scatter(filtered_df,
                                    x='Jauge finale',
                                    y='Taux de remplissage (%)',
                                    size='Jauge finale',
                                    color='Genre',
                                    color_discrete_map=color_map,
                                    custom_data=['event_title', 'billets_valides', 'Jauge finale', 'Taux de remplissage (%)', 'event_start_date'],
                                    title=f'Classement des concerts par jauge et taux de remplissage{" en " + str(selected_year) if selected_year != "Toutes les années" else ""}')
                    
                    fig.update_traces(
                        opacity=0.8,
                        hovertemplate="""
                        <b>%{customdata[0]}</b><br>
                        Date: %{customdata[4]}<br>
                        Billets validés: %{customdata[1]:,.0f}<br>
                        Jauge: %{customdata[2]:,.0f}<br>
                        Taux de remplissage: %{customdata[3]:.1f}%
                        """
                    )
                    
                    fig.update_layout(
                        hoverlabel=dict(
                            font_size=14,
                            align='left'
                        ),
                        plot_bgcolor='white'
                    )
                    
                    jauges_uniques = sorted(filtered_df['Jauge finale'].unique())
                    fig.update_xaxes(tickvals=jauges_uniques, ticktext=jauges_uniques, title_text="Jauge")
                    st.plotly_chart(fig)
                    
                    # Message d'instruction
                    st.info("Découvrez le classement des concerts par taux de remplissage en sélectionnant une jauge👇🏻")
                    
                    # Filtre de jauge et affichage des événements correspondants
                    jauges_uniques = sorted(filtered_df['Jauge finale'].unique())
                    jauges_uniques.insert(0, "Toutes les jauges")
                    selected_jauge = st.selectbox('Jauge', jauges_uniques, key='selected_jauge_occupancy')
                    
                    if selected_jauge != "Toutes les jauges":
                        jauge_filtered_df = filtered_df[filtered_df['Jauge finale'] == selected_jauge]
                        if not jauge_filtered_df.empty:
                            st.markdown(f"#### Événements avec une jauge de {selected_jauge}")
                            events_df = jauge_filtered_df[['event_title', 'event_start_date', 'event_description_clean', 'Taux de remplissage (%)']].drop_duplicates()
                            events_df['event_start_date'] = pd.to_datetime(events_df['event_start_date'])
                            events_df = events_df.sort_values('Taux de remplissage (%)', ascending=False)
                            
                            for i, (_, row) in enumerate(events_df.iterrows(), 1):
                                event_date = row['event_start_date'].strftime('%d/%m/%Y')
                                st.write(f"{i}. {row['event_title']} ({event_date}) - Taux de remplissage : {row['Taux de remplissage (%)']:.1f}%")
                                if pd.notna(row['event_description_clean']):
                                    with st.expander("Voir la description de l'événement"):
                                        st.markdown(row['event_description_clean'])
                
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse du taux de remplissage: {e}")
            elif st.session_state.current_analysis == "tickets":
                try:
                    df = load_data()
                    df['event_start_date'] = pd.to_datetime(df['event_start_date'])
                    
                    # Préparation des données pour tous les genres
                    df_all_genres = df.assign(Genre=df['Genre'].str.split(',')).explode('Genre')
                    df_all_genres['Genre'] = df_all_genres['Genre'].str.strip()
                    
                    # Grouper par année et genre
                    billets_par_annee_genre = df_all_genres.groupby([df_all_genres['event_start_date'].dt.year, 'Genre']).agg({
                        'event_title': 'nunique',
                        'billets_valides': 'size'
                    }).reset_index()
                    billets_par_annee_genre.columns = ['Année', 'Genre', 'Événements uniques', 'Nombre de billets']
                    
                    # Créer le graphique avec sélection de genre intégrée
                    fig = px.line(billets_par_annee_genre,
                                x='Année',
                                y='Nombre de billets',
                                color='Genre',
                                title='Evolution du nombre de billets par genre',
                                markers=True)
                    
                    # Personnaliser l'encadré au survol
                    fig.update_traces(
                        hovertemplate='<span style="font-size: 16px;"><b>Année</b>: %{x}<br><b>Genre</b>: %{customdata[0]}<br><b>Nombre de billets</b>: %{y}<br></span><extra></extra>',
                        customdata=billets_par_annee_genre[['Genre']]
                    )
                    
                    # Personnaliser l'axe x et la mise en page
                    fig.update_xaxes(dtick=1, type='linear')
                    fig.update_layout(
                        hoverlabel=dict(
                            font_size=16,
                            font_family="Arial"
                        ),
                        showlegend=True,
                        legend_title_text='Genres'
                    )
                    
                    # Afficher le graphique
                    st.plotly_chart(fig)
                    
                    # Deuxième graphique - Taux de remplissage par genre
                    df_grouped = df_all_genres.groupby(['Genre', df_all_genres['event_start_date'].dt.year])['Taux de remplissage (%)'].mean().reset_index()
                    df_grouped.columns = ['Genre', 'event_start_date', 'Taux de remplissage (%)']
                    
                    fig2 = px.line(df_grouped,
                                x='event_start_date',
                                y='Taux de remplissage (%)',
                                color='Genre',
                                title='Evolution du taux de remplissage par genre')
                    
                    # Configuration identique de l'axe X pour les deux graphiques
                    fig2.update_xaxes(dtick=1, type='linear')
                    
                    # Mise à jour du template de survol pour le deuxième graphique
                    fig2.update_traces(
                        mode='lines+markers',
                        hovertemplate='<span style="font-size: 16px;"><b>Année</b>: %{x}<br><b>Taux de remplissage</b>: %{y:.1f}%<br></span><extra></extra>',
                        customdata=df_grouped[['Genre']]
                    )
                    
                    fig2.update_layout(
                        xaxis_title="Année",
                        yaxis_title="Taux de remplissage moyen (%)",
                        hovermode='closest',
                        width=800,
                        height=500,
                        hoverlabel=dict(
                            font_size=16,
                            font_family="Arial"
                        )
                    )
                    
                    st.plotly_chart(fig2)
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse de la billetterie: {e}")
            elif st.session_state.current_analysis == "venues":
                try:
                    df = load_data()
                    df['year'] = pd.to_datetime(df['event_start_date']).dt.year
                    years = sorted(df['year'].unique(), reverse=True)
                    years.insert(0, "Toutes les années")
                    
                    # Extraction et nettoyage des genres
                    all_genres = []
                    for genres in df['Genre'].dropna():
                        all_genres.extend([g.strip() for g in genres.split(',')])
                    unique_genres = sorted(list(set(all_genres)))
                    unique_genres.insert(0, "Tous les genres")
                    
                    # Filtres
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_year = st.selectbox('Année', years, key='selected_year_venues')
                    with col2:
                        selected_genre = st.selectbox('Genre', unique_genres, key='selected_genre_venues')
                    
                    # Filtrage des données
                    filtered_df = df.copy()
                    if selected_year != "Toutes les années":
                        filtered_df = filtered_df[filtered_df['year'] == selected_year]
                    if selected_genre != "Tous les genres":
                        filtered_df = filtered_df[filtered_df['Genre'].str.contains(selected_genre, na=False)]
                    
                    # Compter les événements par lieu
                    events_par_lieu = filtered_df.groupby('venue_name')['event_id'].nunique().reset_index()
                    events_par_lieu.columns = ['venue_name', 'nb_events']
                    
                    # Appliquer le géocodage aux adresses uniques
                    venues = filtered_df[['venue_name', 'venue_address']].drop_duplicates()
                    venues[['latitude', 'longitude']] = venues['venue_address'].apply(get_coordinates)
                    
                    # Fusionner avec le nombre d'événements
                    venues = venues.merge(events_par_lieu, on='venue_name', how='left')
                    
                    # Créer une carte avec Plotly
                    fig = px.scatter_mapbox(venues.dropna(),
                                        lat='latitude',
                                        lon='longitude',
                                       hover_name='venue_name',
                                       hover_data={'nb_events': True, 'latitude': False, 'longitude': False},
                                       custom_data=['nb_events'],
                                       color_discrete_sequence=['blue'],
                                       size_max=25)
                    
                    # Configurer la carte Plotly
                    fig.update_layout(
                        mapbox_style='carto-positron',
                        mapbox=dict(
                            center=dict(lat=44.8378, lon=-0.5792),
                            zoom=12
                        ),
                        height=600,
                        hoverlabel=dict(
                            font_size=16,
                            font_family="Arial"
                        )
                    )
                    
                    # Mettre à jour le template de survol
                    fig.update_traces(
                        marker=dict(size=15),
                        hovertemplate="<b>%{hovertext}</b><br>Nombre d'événements: %{customdata[0]}<extra></extra>"
                    )

                    st.plotly_chart(fig)

                    # Message d'instruction
                    st.info("Veuillez sélectionner un lieu pour voir les événements qui y sont rattachés 👇🏻")

                    # Liste des lieux uniques pour le filtre
                    unique_venues = sorted(filtered_df['venue_name'].unique())
                    unique_venues.insert(0, "Tous les lieux")
                    selected_venue = st.selectbox('Lieu', unique_venues, key='selected_venue_filter')

                    # Afficher les événements si un lieu est sélectionné
                    if selected_venue != "Tous les lieux":
                        venue_events = filtered_df[filtered_df['venue_name'] == selected_venue]
                        if not venue_events.empty:
                            st.markdown(f"#### Événements à {selected_venue}")
                            events_df = venue_events[['event_title', 'event_start_date', 'event_description_clean']].drop_duplicates()
                            events_df['event_start_date'] = pd.to_datetime(events_df['event_start_date'])
                            events_df = events_df.sort_values('event_start_date', ascending=True)
                            for i, (_, row) in enumerate(events_df.iterrows(), 1):
                                event_date = row['event_start_date'].strftime('%d/%m/%Y')
                                st.write(f"{i}. {row['event_title']} ({event_date})")
                                if pd.notna(row['event_description_clean']):
                                    with st.expander("Voir la description de l'événement"):
                                        st.markdown(row['event_description_clean'])

                except Exception as e:
                    st.error(f"Erreur lors de l'analyse des lieux: {e}")







elif selected == "Assistant":
    # Configuration de la mise en page
    st.markdown("<h2 style='text-align: center; color: #756AB0;'>🎵 Assistant de recherche de répertoire</h2>", unsafe_allow_html=True)
    

    # Chargement des données
    df = pd.read_csv("https://raw.githubusercontent.com/jordigarciag/projet_3_wild/b4fd7b8e1cf28924e7230a5221c7e8b8cecb38b9/df_final_ml_v9.csv")

    # Fonction pour nettoyer et convertir les années
    def clean_year(year):
        if pd.isna(year):
            return None
        if isinstance(year, str):
            year = ''.join(filter(str.isdigit, year))
        return int(year) if year else None

    # Nettoyer les colonnes d'années
    df['annee_naissance'] = df['annee_naissance'].apply(clean_year)
    df['annee_deces'] = df['annee_deces'].apply(clean_year)

    # Appliquer get_dummies pour les variables catégorielles
    df_encoded = pd.get_dummies(df, columns=['nationalite', 'genre', 'sexe'])

    # Obtenir les années minimales en excluant les valeurs None
    min_annee_deces = int(df['annee_deces'].dropna().min())
    min_annee_naissance = int(df['annee_naissance'].dropna().min())

    # Initialisation des variables de session
    if 'compositeur_search' not in st.session_state:
        st.session_state.compositeur_search = ""
    if 'year_type' not in st.session_state:
        st.session_state.year_type = "Aucune"
    if 'nationalities' not in st.session_state:
        st.session_state.nationalities = []
    if 'genres' not in st.session_state:
        st.session_state.genres = []
    if 'sexes' not in st.session_state:
        st.session_state.sexes = []

    # Création de la mise en page principale avec colonnes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Choisissez votre méthode de recherche')
        tab1, tab2 = st.tabs(["🎯 Recherche par compositeur similaire", "🔍 Recherche par critères"])

        with tab1:
            st.write("Trouvez des compositeurs similaires en saisissant le nom d'un compositeur que vous connaissez déjà.")
            compositeur_search = st.text_input("Rechercher un compositeur", value=st.session_state.compositeur_search)
            
            if compositeur_search:
                compositeurs_filtres = df[df['compositeur'].str.contains(compositeur_search, case=False, na=False)]['compositeur'].unique()
                if len(compositeurs_filtres) > 0:
                    compositeur_selected = st.selectbox("Compositeurs trouvés", compositeurs_filtres)
                else:
                    st.warning("Aucun compositeur trouvé")

        with tab2:
            st.write("Trouvez des compositeurs en spécifiant vos critères de recherche.")
            
            year_type = st.radio(
                "Choisir le type d'année",
                ["Aucune", "Année de naissance", "Année de décès"],
                index=["Aucune", "Année de naissance", "Année de décès"].index(st.session_state.year_type)
            )
            
            if year_type == "Année de naissance":
                birth_year = st.slider("Année de naissance du compositeur",
                                 min_annee_naissance,
                                 int(df['annee_naissance'].dropna().max()),
                                 int(df['annee_naissance'].dropna().mean()))
            elif year_type == "Année de décès":
                death_year = st.slider("Année de décès du compositeur",
                                 min_annee_deces,
                                 int(df['annee_deces'].dropna().max()),
                                 int(df['annee_deces'].dropna().mean()))

            nationalities = st.multiselect(
                "Nationalité(s)",
                options=sorted(df['nationalite'].unique().tolist()),
                default=st.session_state.nationalities
            )

            genres = st.multiselect(
                "Genre(s) musical(aux)",
                options=sorted(df['genre'].unique().tolist()),
                default=st.session_state.genres
            )

            sexes = st.multiselect(
                "Sexe",
                options=sorted(df['sexe'].unique().tolist()),
                default=st.session_state.sexes
            )

        # Boutons côte à côte
        search_col1, search_col2 = st.columns(2)
        with search_col1:
            search_button = st.button('🔍 Rechercher')
        with search_col2:
            clear_button = st.button('🔄 Effacer tous les filtres')
            if clear_button:
                st.session_state.compositeur_search = ""
                st.session_state.year_type = "Aucune"
                st.session_state.nationalities = []
                st.session_state.genres = []
                st.session_state.sexes = []
                st.rerun()

    # Colonne des résultats (à droite)
    with col2:
        st.subheader('Résultats')
        if search_button:
            # Préparation des données pour le modèle KNN
            features = []
            feature_columns = []

            if compositeur_search and 'compositeur_selected' in locals():
                composer_data = df[df['compositeur'] == compositeur_selected].iloc[0]
                query_features = []
                
                if pd.notna(composer_data['annee_naissance']):
                    query_features.append(composer_data['annee_naissance'])
                    feature_columns.append('annee_naissance')
                if pd.notna(composer_data['annee_deces']):
                    query_features.append(composer_data['annee_deces'])
                    feature_columns.append('annee_deces')
                
                mask = df['compositeur'] != compositeur_selected
            else:
                mask = pd.Series(True, index=df.index)
                
                if year_type == "Année de naissance":
                    query_features = [birth_year]
                    feature_columns = ['annee_naissance']
                elif year_type == "Année de décès":
                    query_features = [death_year]
                    feature_columns = ['annee_deces']
                else:
                    query_features = []
                
                if nationalities:
                    mask &= df['nationalite'].isin(nationalities)
                if genres:
                    mask &= df['genre'].isin(genres)
                if sexes:
                    mask &= df['sexe'].isin(sexes)

            filtered_df = df[mask]
            filtered_df_encoded = df_encoded[mask]

            if filtered_df.empty:
                st.warning("Aucun compositeur ne correspond à vos critères.")
            else:
                if feature_columns:
                    mask_valid = filtered_df[feature_columns].notna().all(axis=1)
                    filtered_df = filtered_df[mask_valid]
                    filtered_df_encoded = filtered_df_encoded[mask_valid]

                    if filtered_df.empty:
                        st.warning("Aucun compositeur ne correspond à vos critères après filtrage des dates.")
                    else:
                        X = filtered_df[feature_columns].values
                        encoded_columns = [col for col in filtered_df_encoded.columns if col.startswith(('nationalite_', 'genre_', 'sexe_'))]
                        X = np.hstack((X, filtered_df_encoded[encoded_columns].values))

                        scaler = MinMaxScaler()
                        X_scaled = scaler.fit_transform(X)

                        knn = NearestNeighbors(n_neighbors=min(5, len(X_scaled)))
                        knn.fit(X_scaled)

                        query_point = np.zeros(X.shape[1])
                        query_point[:len(query_features)] = query_features

                        if compositeur_search and 'compositeur_selected' in locals():
                            composer_encoded = df_encoded[df_encoded.index == composer_data.name][encoded_columns].values[0]
                            query_point[len(query_features):] = composer_encoded
                        else:
                            if nationalities or genres or sexes:
                                for col_name in encoded_columns:
                                    if col_name.startswith('nationalite_') and nationalities:
                                        nat = col_name.replace('nationalite_', '')
                                        if nat in nationalities:
                                            idx = len(query_features) + encoded_columns.index(col_name)
                                            query_point[idx] = 1
                                    elif col_name.startswith('genre_') and genres:
                                        genre = col_name.replace('genre_', '')
                                        if genre in genres:
                                            idx = len(query_features) + encoded_columns.index(col_name)
                                            query_point[idx] = 1
                                    elif col_name.startswith('sexe_') and sexes:
                                        sexe = col_name.replace('sexe_', '')
                                        if sexe in sexes:
                                            idx = len(query_features) + encoded_columns.index(col_name)
                                            query_point[idx] = 1

                        query_point = query_point.reshape(1, -1)
                        query_point_scaled = scaler.transform(query_point)

                        distances, indices = knn.kneighbors(query_point_scaled)

                        for i, idx in enumerate(indices[0]):
                            composer = filtered_df.iloc[idx]
                            st.write(f"🎼 {composer['compositeur']} - {composer['oeuvre']}")
                            st.write(f"🎂 Naissance: {int(composer['annee_naissance']) if pd.notna(composer['annee_naissance']) else 'Non disponible'}")
                            st.write(f"✝️ Décès: {int(composer['annee_deces']) if pd.notna(composer['annee_deces']) else 'Non disponible'}")
                            st.write(f"🌍 Nationalité: {composer['nationalite']}")
                            st.write(f"🎵 Genre musical: {composer['genre']}")
                            st.write(f"⚥ Sexe: {composer['sexe']}")
                            st.write("---")
                else:
                    for _, composer in filtered_df.head(5).iterrows():
                        st.write(f"🎼 {composer['compositeur']} - {composer['oeuvre']}")
                        st.write(f"🎂 Naissance: {int(composer['annee_naissance']) if pd.notna(composer['annee_naissance']) else 'Non disponible'}")
                        st.write(f"✝️ Décès: {int(composer['annee_deces']) if pd.notna(composer['annee_deces']) else 'Non disponible'}")
                        st.write(f"🌍 Nationalité: {composer['nationalite']}")
                        st.write(f"🎵 Genre musical: {composer['genre']}")
                        st.write(f"⚥ Sexe: {composer['sexe']}")
                        st.write("---")

    # Mise à jour des variables de session
    st.session_state.compositeur_search = compositeur_search
    st.session_state.year_type = year_type
    st.session_state.nationalities = nationalities
    st.session_state.genres = genres
    st.session_state.sexes = sexes



elif selected == "Suggestions":
    # Configuration de la mise en page
    st.markdown("<h2 style='text-align: center; color: #756AB0;'>💡 Suggestions pour les saisons à venir</h2>", unsafe_allow_html=True)
    st.write("")
    
    # Création de deux colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("https://cathedra.fr/wp-content/uploads/2024/12/cathedra_4266985_731191115719cathedra_5_515995cathedra_00689830882_n-min.jpeg")
        st.image("https://cathedra.fr/wp-content/uploads/2024/12/Concert-lHomme-arme-larmes-et-alarmes-Notre-Dame-Bordeaux-04-min-min.jpg")
        st.image("https://cathedra.fr/wp-content/uploads/2024/12/cathedra_3773724_731196319052208_5102679244012699935_n-min.jpeg")
        st.image("https://cathedra.fr/wp-content/uploads/2024/12/DSC_3923-min-min.jpg")
        st.image("https://cathedra.fr/wp-content/uploads/2024/12/Z6Z_0168-min-min.jpg")
    
    with col2:
        st.markdown("""
        ### 🔍 Synthèse
        
        L'analyse des données de Cathedra révèle plusieurs catégories de concerts particulièrement plébiscitées par le public :

        🎭 **Concerts participatifs**
        
        Les événements comme "Jubilate Deo" et la "Messe du couronnement", qui concluent les stages annuels de Cathedra, connaissent un grand succès.

        🏰 **Patrimoine bordelais**
        
        Les concerts mettant en valeur l'héritage local, notamment à travers :
        
        • Les œuvres de Clément Janequin (concert "Si François 1er m'était conté")
        
        • Les prestations des grands chœurs bordelais, en particulier la Maîtrise de Bordeaux interprétant Emmanuel Filet et Alexis Duffaure

        ⭐ **Artistes prestigieux**
        
        Les concerts mettant en scène des personnalités reconnues :
        
        • Le ténor Stanislas de Barbeyrac (Concert d'ouverture 2017)
        
        • Le claveciniste William Christie (Les Arts Florissants)
        
        • Le violoniste Renaud Capuçon

        🎼 **Grands classiques**
        
        Les œuvres des compositeurs les plus connus du grand public (Bach, Fauré, Mozart)

        🎵 **Musique baroque** :
        
        • Baroque français ("Dialogues Sacrés" avec le Centre de Musique Baroque de Versailles)
        
        • Baroque d'ailleurs (concert "Bach/Corelli")

        Les genres musicaux obtenant les meilleurs taux de remplissage se classent dans l'ordre suivant : Moderne/Contemporain, Classique, Post-romantique et Baroque.

        Cette diversité dans l'offre musicale apparaît comme un élément clé pour maintenir la fidélité du public existant tout en attirant de nouveaux spectateurs.

        ---
        
        
        ### 💡 Programmation
        
        Sur la base de cette analyse et grâce à notre assistant développé avec un modèle de machine learning KNN, nous proposons les suggestions suivantes pour les prochaines saisons, visant à optimiser l'attractivité des concerts tout en préservant l'identité artistique de Cathedra :

        🎭 **Concerts participatifs**
        
        • "Requiem" (Mozart) - Une œuvre majeure de musique sacrée
        
        • "Misa Tango" (Martín Palmeri) - Une fusion innovante entre messe traditionnelle et tango argentin

        🏰 **Héritage bordelais**
        
        • "Œuvres sacrées" de Joseph Valette de Montigny - Maître de chapelle à Saint-André au XVIIIe siècle
        
        • "Anthologie de la Cathédrale" - Un voyage musical à travers l'histoire des compositeurs en lien avec la cathédrale Saint-André

        ⭐ **Artistes de renom**
        
        • Philippe Jaroussky - Contre-ténor de renommée internationale
        
        • Axelle Saint-Cirel - Soprano révélée aux JO de Paris 2024

        🎼 **Grands compositeurs**
        
        • "Messie" (Haendel) - Oeuvre magistrale reconnue pour son impact durable sur le public
        
        • "Messe en si mineur" (Bach) - Monument du répertoire choral

        🎵 **Musique baroque**
        
        • "Grands motets" (Lully/Charpentier) - L'âge d'or de la musique française
        
        • "Cantata Lamento della Vergine" (Antonia Bembo) - Compositrice italienne du XVIIe siècle
        
        • "L'Eraclito Amoroso" (Barbara Strozzi) - Cantate sur l'amour et la mélancolie

        🎹 **Contemporain**
        
        • "Night Ferry" (Anna Clyne) - Œuvre orchestrale évocatrice d'un voyage nocturne en mer, inspirée par Schumann
        """)