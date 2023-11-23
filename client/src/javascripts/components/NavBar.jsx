import React from "react";
import { styles } from "../styles"
import logo from "../../assets/logo1.png"

const Navbar = () => {
      return (
      <nav
        className={`${ styles.paddingX } w-full flex items-center py-5 transition-all fixed top-0 z-20 bg-colour`} >

        <div className='w-full flex justify-between items-center max-w-7xl mx-auto '>
            <div className="flex items-center gap-2">

              <img src={logo} alt='logo' className='w-16 h-16 object-contain' />
              <p className='text-white text-5xl font-light cursor-pointer flex font-montserrat' style={{ fontSize: '18px' }}>
                VocalVerify &nbsp;
                <span className='sm:block hidden'> | DeepFake Test</span>
              </p>
            </div>
        </div>
      </nav>
    );
  };
  
  export default Navbar;