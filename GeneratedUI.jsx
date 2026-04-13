```jsx
import React from 'react';

const Button = () => {
  return (
    <div className="flex justify-start items-start w-100 h-100 bg-gray-200">
      <button className="text-lg text-gray-700 hover:text-gray-900 absolute top-0 left-0" type="text">
        Click Me Button
      </button>
    </div>
  );
};

const App = () => {
  return (
    <div className="h-screen w-screen flex justify-center items-center">
      <Button />
    </div>
  );
};

export default App;
```