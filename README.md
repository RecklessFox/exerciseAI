Web server running a song and album cover generator APIs in the genre of Taylor Swift

build docker image with Dockerfile:  
    ```docker build -t songgenerator .```

run web server on port 8000 with:  
    ```docker-compose up```

go to **localhost:8000** (ideally on a web browser) and follow instructions on how to use the web server  

endpoints:  
    **localhost:8000** -> home  
    **localhost:8000/inferText/{input_lyrics}** -> text(song) generator using fine tuned (with lyrics from the dataset) GPT2 model.  
    **localhost:8000/inferImage/{input_album_theme}** -> image(album cover) generator using stable diffusion 1-5 model.  
