import React from 'react';
import Work from "./components/Work.jsx";
import Navbar from './components/NavBar.jsx';
// import FloatingObjects from './FloatingObjects';

const App = () => {
  return (
    <div>
      <div className='relative z-0 bg-primary bg-transparent'>

        <div className='bg-heo-pattern bg-cover bg-no-repeat bg-center'>
          <Navbar />
          {/* <Hero /> */}
        </div>
        <Work/>
        {/* <FloatingObjects/> */}
      </div>

    </div>
  );
}

export default App;
