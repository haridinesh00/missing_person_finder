import { createContext, useState } from "react";

export const Cardcontext=createContext(null)

export default function Context({children}){
    const[card,setCard]=useState(null);
    
    return(
        
        <Cardcontext.Provider value={{card,setCard}}>
            {children}
        </Cardcontext.Provider>
        
    
    )
}