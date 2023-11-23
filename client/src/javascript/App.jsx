import '../css/App.css';
import HomeScreen from './components/HomeScreen';
import NavBar from './components/NavBar';
import Floating_objects from './Floating_objects';

import React from "react";


function App() {
  return (
    <div className="App" >
      <Floating_objects className="z-10"/>
      <NavBar />
      <HomeScreen />
    </div>
  );
}

export default App;