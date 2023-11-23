import React from 'react';
import Navbar from './components/NavBar.jsx';
import Hero from './components/Hero.jsx';
import About from './components/About.jsx';
// import FloatingObjects from './FloatingObjects';

const App = () => {
  return (
    <div>
      <div className='relative z-0 bg-primary bg-transparent'>

        <div className='bg-heo-pattern bg-cover bg-no-repeat bg-center'>
          <Navbar />
          <Hero />
        </div>
        <About />
        {/* <Work/> */}
        {/* <FloatingObjects/> */}
      </div>

    </div>
  );
}

export default App;
