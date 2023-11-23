import React, { useState, useEffect } from "react";
import {Tilt} from 'react-tilt'
import {motion} from 'framer-motion';

import {styles} from '../styles';
import { services } from '../../constants';
import {fadeIn, textVariant} from '../../utils/motion'
import { SectionWrapper } from '../../hoc';

const ServiceCard = ({ index, images }) => (
  <Tilt className='xs:w-full w-full'>
    <motion.div
      variants={fadeIn("right", "spring", index * 0.5, 0.75)}
      className='w-full red-gradient p-[1px] rounded-[20px] shadow-card'
    >
      <div
        options={{
          max: 45,
          scale: 1,
          speed: 450,
        }}
        className='bg-primary rounded-[20px] py-5 px-12 min-h-[280px] flex justify-evenly items-center flex-col'
      >
        {images && images.length > 0 ? (
          images.map((image, i) => (
            <img
              key={i}
              src={`data:image/png;base64,${image}`}
              alt={`Image ${i}`}
            />
          ))
        ) : (
          <p className ={styles.sectionSubText} >Oppsiies Nothing to show yet (ᗒᗣᗕ)՞</p>
        )}
      </div>
    </motion.div>
  </Tilt>
);

const About = () => {
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
    <>
    <motion.div variants={textVariant()}>
      <div className ={styles.sectionSubText}>
          {typeof test_data.test === "undefined" ? (
              <p className ={styles.sectionSubText} >Waiting for you to upload the audio file (づ ◕‿◕ )づ</p> ) : (
            <div>
              File Uploaded ٩(◕‿◕)۶
              </div>
            )}
        </div>
        
      <div className ={styles.sectionHeadText}>
        {typeof test_data.test === "undefined" ? (
            <div className ={styles.sectionHeadText} >No File Uploaded (Θ︹Θ)ს</div> ) : (
          <div>
            Here's the Result ٩(◕‿◕)۶: 
            </div>
          )}
      </div>  

    </motion.div>

    {/* <motion.p 
      variants={fadeIn("","",0.1,1)}
      className="mt-4 text-red-50 text-[17px] max-w-3xl leading-[30px]">
        <div>
            {typeof intro_data.intro === "undefined" ? (
              <p className ={styles.sectionSubText} >loading...</p> ) : (
              <div >
                <div className="flex flex-row">
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
    </motion.p> */}

    <motion.p 
        variants={fadeIn("","",0.1,1)}
        className="mt-4 text-red-50 text-[17px] max-w-3xl leading-[30px]">
          <div>
            <input type="file" accept=".wav" onChange={handleFileChange} />
            {/* <button onClick={handleUpload}>Upload</button> */}
            <button onClick={handleUpload} className="bg-red-600 py-3 text-white rounded-md w-3/12 mt-10 hover:bg-red-700 transition">
              Upload
            </button>
            {audioFile && <audio controls src={audioFile} />}
          </div>
          <div>
            {typeof test_data.test === "undefined" ? (
               <p className ={styles.sectionSubText} ></p> ) : (
              <div >
                  {test_data.test.map((intro, i) => (
                    <div className="flex flex-row">
                    <p key={i}>{`> ${intro}`}</p>
                    </div>
                  ))}
                  {/* {test_data.image.map((image, i) => (
                    <img
                      key={i}
                      src={`data:image/png;base64,${image}`}
                      alt={`Image ${i}`}
                    />
                  ))} */}
                  {test_data.result.map((result, i) => (
                    <p key={i}>{result}</p>
                  ))}
                </div>
            )}
          </div>
      </motion.p>

      <div className='mt-20 flex flex-wrap gap-10'>
        {services.map((service, index) => (
          <ServiceCard
            key={service.title}
            index={index}
            {...service}
            images={test_data.image} // Pass the images here
          />
        ))}
      </div>

      {/* <div className='mt-20'>
        {typeof test_data.test === "undefined" ? (
          <div></div>
        ) : (
          services.map((service, index) => (
            <ServiceCard
              key={service.title}
              index={index}
              {...service}
              images={test_data.image}
            />
          ))
        )}
      </div> */}
    </>
  )
}

export default SectionWrapper(About,"about")