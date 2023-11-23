import React, { useState, useEffect } from "react";
function HomeScreen() {
  const [intro_data, setData] = useState([{}]);
  const [test_data, settestData] = useState([{}]);
  const [buttonClicked, setButtonClicked] = useState(false);
  
  useEffect(() => {
    fetch("/intro")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! Status: ${res.status}`);
        }
        return res.json();
      })
      .then((introData) => {
        if (introData) {
          setData(introData);
          console.log(introData);
        } else {
          console.error("Empty response received");
        }
      })
      .catch((error) => {
        console.error("Error fetching or parsing data:", error);
      });
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("/test");
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const testdata = await response.json();
        settestData(testdata);
        console.log(testdata);
      } catch (error) {
        console.error("Error fetching data:", error.message);
        // Handle the error as needed
      }
    };

    if (buttonClicked) {
      fetchData(); // Call the async function when the button is clicked
    }
  }, [buttonClicked]);
  const [audioFile, setAudioFile] = useState(null);
  const [file, setFile] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setAudioFile(URL.createObjectURL(file));
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
  };

  const handleUpload = async () => {
    try {
      if (!file) {
        console.error("No file selected.");
        return;
      }

      const formData = new FormData();
      formData.append("audioFile", file);

      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      // Handle the response from the backend as needed
      console.log("Upload successful", response);
      setButtonClicked(true);
    } catch (error) {
      console.error("Error uploading file", error);
    }
  };

  return (
    <div>
      <div>
        <h1> this is how site works - sample audio h for info purpose </h1>
        <div>
          {typeof intro_data.intro === "undefined" ? (
            <p>loading...</p>
          ) : (
            <div>
              <div>
                {intro_data.intro.map((intro, i) => (
                  <p key={i}>{intro}</p>
                ))}
                {intro_data.image.map((image, i) => (
                  <img
                    key={i}
                    src={`data:image/png;base64,${image}`}
                    alt={`Image ${i}`}
                  />
                ))}
                {intro_data.result.map((result, i) => (
                  <p key={i}>{result}</p>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
      <div>
        <h1>yaha p test krne wali file upload kro</h1>
        <input type="file" accept=".wav" onChange={handleFileChange} />
        <button onClick={handleUpload}>Upload</button>
        {audioFile && <audio controls src={audioFile} />}
      </div>
      <div>
        {typeof test_data.test === "undefined" ? (
          <p>loading...</p>
        ) : (
          <div>
            <div>
              <h1>ab yaha p model predict krega</h1>
              {test_data.test.map((intro, i) => (
                <p key={i}>{intro}</p>
              ))}
              {test_data.image.map((image, i) => (
                <img
                  key={i}
                  src={`data:image/png;base64,${image}`}
                  alt={`Image ${i}`}
                />
              ))}
              {test_data.result.map((result, i) => (
                <p key={i}>{result}</p>
              ))}
            </div>
          </div>
        )}
      </div>
      <div className="relative h-full w-full">
            
            <div className="bg-black w-full h-full lg:bg-opacity-50">

                {/* <nav className="px-12 py-5">
                    <img src="/images/logo.png" alt="Logo" className="h-12"/>
                </nav> */}

                <div className="flex justify-center">

                <div className="bg-black bg-opacity-70 px-16 py-16 self-center mt-2 lg:w-2/5 lg:max-w-md rounded-md w-full">


                    <button onClick={handleUpload} className="bg-red-600 py-3 text-white rounded-md w-full mt-10 hover:bg-red-700 transition">
                        Upload
                    </button>
                    
                </div>
                </div>
            </div>
        </div>
    </div>
  );
}

export default HomeScreen;
