import React from 'react'
import {Tilt} from 'react-tilt'
import {motion} from 'framer-motion';

import {styles} from '../styles';
import { services } from '../../constants';
import {fadeIn, textVariant} from '../../utils/motion'
import { SectionWrapper } from '../../hoc';

const ServiceCard = ({ index, title, icon }) => (
  <Tilt className='xs:w-[250px] w-full'>
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
        {/* <img
          src={icon}
          alt='web-development'
          className='w-30 h-30 object-contain'
        /> */}

        <h3 className='text-white text-[20px] font-bold text-center'>
          {title}
        </h3>
      </div>
    </motion.div>
  </Tilt>
);


const About = () => {
  return (
    <>

    <motion.div variants={textVariant()}>
      <p className ={styles.sectionSubText}>
        Introduction</p>

      <h2 className={styles.sectionHeadText} >
        Overview </h2>

    </motion.div>

    <motion.p 
      variants={fadeIn("","",0.1,1)}
      className="mt-4 text-red-50 text-[17px] max-w-3xl leading-[30px]">
        Hi, I'm Mahansh Aditya, a sophomore studying Computer Science and Design at IIIT-Delhi. My passion lies in software development, and I aspire to become a full stack developer. 
    </motion.p>

    <motion.p 
        variants={fadeIn("", "", 0.2, 1)}
        className="mt-4 text-red-50 text-[17px] max-w-3xl leading-[30px] mb-8">
        I have experience in programming languages such as Java, Python, C, C++ , MSQL and JavaScript. Although I have not yet participated in hackathons or coding competitions, I am constantly working on personal projects to improve my abilities and expand my knowledge.
      </motion.p>

      <motion.p 
        variants={fadeIn("", "", 0.2, 1)}
        className="mt-4 text-red-50 text-[17px] max-w-3xl leading-[30px] mb-8">
          I am a detail-oriented individual and enjoy collaborating with others on group projects. I am excited to continue developing my skills and pursuing opportunities in the software development field.
      </motion.p>

    <div className='mt-20 flex flex-wrap gap-10'>

      {services.map((services, index) => (
        <ServiceCard key={services.title} index={index}{...services}/>
      ))}
    </div>
    </>
  )
}

export default SectionWrapper(About,"about")