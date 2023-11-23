import React, {useState, useEffect } from "react";
import logo from "../../assets/logo.png"

const TOP_OFFSET = 66;
const NavBar = ()=>{
    const[showBackground, setShowBackground] = useState(false);

    useEffect(() =>{
        const handleScroll = () =>{
            if(window.scrollY >= TOP_OFFSET){
                setShowBackground(true);
            }
            else{
                setShowBackground(false);
            }
        }

        window.addEventListener('scroll', handleScroll);

        return () =>{
            window.removeEventListener('scroll', handleScroll);
        }
    },[]);

    return(
        <div className="w-full fixed z-40">
            <div
            className=
            {`px-4
            md:px-16
            py-6
            flex
            flex-row
            items-center
            transition
            duration-1000
            ${showBackground ? 'bg-zinc-900 bg-opacity-90' : ''}
            `}>
                <img className="h-4 lg:h-7 "  src={logo} alt='Logo'/>

                <div className="
                flex-row
                ml-8
                gap-7
                hidden
                lg:flex
                ">

                </div>

                <div className="flex flex-row ml-auto gap-7 items-center">

                    <div className="flex flex-row items-center gap-2 cursor-pointer relative">

                        <div className="w-10 h-10 lg:w-16 lg:h-16 overflow-hidden">
                            <img src='/images/account_avatar.png' alt='Profile'/>
                        </div>

                    </div>

                </div>
            </div>
        </div>
    )
}

export default NavBar;